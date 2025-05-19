import torch
import torch.nn as nn
from torch.nn import functional as F

try:
    import torch.distributed.nn
    from torch import distributed as dist
    has_distributed = True
except ImportError:
    has_distributed = False

try:
    import horovod.torch as hvd
except ImportError:
    hvd = None


def gather_features(image_features,
                    text_features,
                    local_loss=False,
                    gather_with_grad=False,
                    rank=0,
                    world_size=1,
                    use_horovod=False):
    assert has_distributed, 'torch.distributed did not import correctly, please use a PyTorch version with support.'
    if use_horovod:
        assert hvd is not None, 'Please install horovod'
        if gather_with_grad:
            all_image_features = hvd.allgather(image_features)
            all_text_features = hvd.allgather(text_features)
        else:
            with torch.no_grad():
                all_image_features = hvd.allgather(image_features)
                all_text_features = hvd.allgather(text_features)
            if not local_loss:
                # ensure grads for local rank when all_* features don't have a gradient
                gathered_image_features = list(
                    all_image_features.chunk(world_size, dim=0))
                gathered_text_features = list(
                    all_text_features.chunk(world_size, dim=0))
                gathered_image_features[rank] = image_features
                gathered_text_features[rank] = text_features
                all_image_features = torch.cat(gathered_image_features, dim=0)
                all_text_features = torch.cat(gathered_text_features, dim=0)
    else:
        # We gather tensors from all gpus
        if gather_with_grad:
            all_image_features = torch.cat(
                torch.distributed.nn.all_gather(image_features), dim=0)
            all_text_features = torch.cat(
                torch.distributed.nn.all_gather(text_features), dim=0)
        else:
            gathered_image_features = [
                torch.zeros_like(image_features) for _ in range(world_size)
            ]
            gathered_text_features = [
                torch.zeros_like(text_features) for _ in range(world_size)
            ]
            dist.all_gather(gathered_image_features, image_features)
            dist.all_gather(gathered_text_features, text_features)
            if not local_loss:
                # ensure grads for local rank when all_* features don't have a gradient
                gathered_image_features[rank] = image_features
                gathered_text_features[rank] = text_features
            all_image_features = torch.cat(gathered_image_features, dim=0)
            all_text_features = torch.cat(gathered_text_features, dim=0)

    return all_image_features, all_text_features


class ClipLoss(nn.Module):
    def __init__(
        self,
        local_loss=False,
        gather_with_grad=False,
        cache_labels=False,
        rank=0,
        world_size=1,
        use_horovod=False,
    ):
        super().__init__()
        self.local_loss = local_loss
        self.gather_with_grad = gather_with_grad
        self.cache_labels = cache_labels
        self.rank = rank
        self.world_size = world_size
        self.use_horovod = use_horovod

        # cache state
        self.prev_num_logits = 0
        self.labels = {}

    def forward(self,
                image_features,
                text_features,
                logit_scale,
                ground_labels=None,
                ignore=False,
                google_sup_loss=False):
        assert not (ignore and google_sup_loss), 'please specify only one'
        device = image_features.device
        if self.world_size > 1:
            all_image_features, all_text_features = gather_features(
                image_features, text_features, self.local_loss,
                self.gather_with_grad, self.rank, self.world_size,
                self.use_horovod)

            if self.local_loss:
                logits_per_image = logit_scale * image_features @ all_text_features.T
                logits_per_text = logit_scale * text_features @ all_image_features.T
            else:
                logits_per_image = logit_scale * all_image_features @ all_text_features.T
                logits_per_text = logits_per_image.T
        else:
            # import pdb;pdb.set_trace()
            logits_per_image = logit_scale * image_features @ text_features.T
            logits_per_text = logit_scale * text_features @ image_features.T

        # calculated ground-truth and cache if enabled
        num_logits = logits_per_image.shape[0]

        if ground_labels is not None:
            ground_labels_repeated = ground_labels.view(1, -1).repeat(
                image_features.shape[0], 1)
            equal_labels = (ground_labels_repeated == ground_labels.view(
                -1, 1)).type(torch.float)
            # equal_labels = torch.eye(equal_labels.shape[0],
            #                          device=device,
            #                          dtype=torch.float)

            if ignore:
                I = torch.eye(equal_labels.shape[0],
                              device=device,
                              dtype=torch.float)
                labels = I - 100 * (equal_labels - I)

                image_logit_exp = torch.exp(
                    logits_per_image -
                    torch.max(logits_per_image, dim=1, keepdim=True).values)
                text_logit_exp = torch.exp(
                    logits_per_text -
                    torch.max(logits_per_text, dim=1, keepdim=True).values)

                image_logit_exp = image_logit_exp * (labels != -100)
                text_logit_exp = text_logit_exp * (labels != -100)

                image_logit_exp = torch.diagonal(image_logit_exp) / torch.sum(
                    image_logit_exp, dim=1)
                text_logit_exp = torch.diagonal(text_logit_exp) / torch.sum(
                    text_logit_exp, dim=1)

                image_logit_exp = -torch.log(image_logit_exp)
                text_logit_exp = -torch.log(text_logit_exp)

                total_loss = torch.mean(image_logit_exp) + torch.mean(
                    text_logit_exp)

                total_loss /= 2
            elif google_sup_loss:
                image_logit_exp = torch.exp(
                    logits_per_image -
                    torch.max(logits_per_image, dim=1, keepdim=True).values)
                image_sum = torch.sum(image_logit_exp, dim=1, keepdim=True)
                image_sum = image_sum.repeat(1, image_logit_exp.shape[1])
                image_sum_sub = image_sum - image_logit_exp
                image_logit_exp /= image_sum_sub
                image_logit_exp = -torch.log(image_logit_exp)
                image_logit_exp *= equal_labels
                loss1 = torch.sum(image_logit_exp, dim=1) / torch.sum(
                    equal_labels, dim=1)
                loss1 = torch.mean(loss1)

                text_logit_exp = torch.exp(
                    logits_per_text -
                    torch.max(logits_per_text, dim=1, keepdim=True).values)
                text_sum = torch.sum(text_logit_exp, dim=1, keepdim=True)
                text_sum = text_sum.repeat(1, text_logit_exp.shape[1])
                text_sum_sub = text_sum - text_logit_exp
                text_logit_exp /= text_sum_sub
                text_logit_exp = -torch.log(text_logit_exp)
                text_logit_exp *= equal_labels
                loss2 = torch.sum(text_logit_exp, dim=1) / torch.sum(
                    equal_labels, dim=1)
                loss2 = torch.mean(loss2)

                total_loss = (loss1 + loss2) / 2
            else:
                labels = equal_labels / torch.sum(equal_labels, dim=1).view(
                    -1, 1)
                total_loss = (F.cross_entropy(logits_per_image, labels) +
                              F.cross_entropy(logits_per_text, labels)) / 2

        else:
            if self.prev_num_logits != num_logits or device not in self.labels:
                labels = torch.arange(num_logits,
                                      device=device,
                                      dtype=torch.long)

                if self.world_size > 1 and self.local_loss:
                    labels = labels + num_logits * self.rank
                if self.cache_labels:
                    self.labels[device] = labels
                    self.prev_num_logits = num_logits
            else:
                labels = self.labels[device]

            total_loss = (F.cross_entropy(logits_per_image, labels) +
                          F.cross_entropy(logits_per_text, labels)) / 2
        return total_loss

def neighbour_exchange(from_rank, to_rank, tensor, group=None):
    tensor_recv = torch.zeros_like(tensor)
    send_op = torch.distributed.P2POp(
        torch.distributed.isend,
        tensor,
        to_rank,
        group=group,
    )
    recv_op = torch.distributed.P2POp(
        torch.distributed.irecv,
        tensor_recv,
        from_rank,
        group=group,
    )
    reqs = torch.distributed.batch_isend_irecv([send_op, recv_op])
    for req in reqs:
        req.wait()
    return tensor_recv


def neighbour_exchange_bidir(left_rank, right_rank, tensor_to_left, tensor_to_right, group=None):
    tensor_from_left = torch.zeros_like(tensor_to_right)
    tensor_from_right = torch.zeros_like(tensor_to_left)
    send_op_left = torch.distributed.P2POp(
        torch.distributed.isend,
        tensor_to_left,
        left_rank,
        group=group,
    )
    send_op_right = torch.distributed.P2POp(
        torch.distributed.isend,
        tensor_to_right,
        right_rank,
        group=group,
    )
    recv_op_left = torch.distributed.P2POp(
        torch.distributed.irecv,
        tensor_from_left,
        left_rank,
        group=group,
    )
    recv_op_right = torch.distributed.P2POp(
        torch.distributed.irecv,
        tensor_from_right,
        right_rank,
        group=group,
    )
    reqs = torch.distributed.batch_isend_irecv([send_op_right, send_op_left, recv_op_right, recv_op_left])
    for req in reqs:
        req.wait()
    return tensor_from_right, tensor_from_left


class NeighbourExchange(torch.autograd.Function):
    @staticmethod
    def forward(ctx, from_rank, to_rank, group, tensor):
        ctx.group = group
        ctx.from_rank = from_rank
        ctx.to_rank = to_rank
        return neighbour_exchange(from_rank, to_rank, tensor, group=group)

    @staticmethod
    def backward(ctx, grad_output):
        return (None, None, None) + (NeighbourExchange.apply(ctx.to_rank, ctx.from_rank, ctx.group, grad_output),)


def neighbour_exchange_with_grad(from_rank, to_rank, tensor, group=None):
    return NeighbourExchange.apply(from_rank, to_rank, group, tensor)


class NeighbourExchangeBidir(torch.autograd.Function):
    @staticmethod
    def forward(ctx, left_rank, right_rank, group, tensor_to_left, tensor_to_right):
        ctx.group = group
        ctx.left_rank = left_rank
        ctx.right_rank = right_rank
        return neighbour_exchange_bidir(left_rank, right_rank, tensor_to_left, tensor_to_right, group=group)

    @staticmethod
    def backward(ctx, *grad_outputs):
        return (None, None, None) + \
            NeighbourExchangeBidir.apply(ctx.right_rank, ctx.left_rank, ctx.group, *grad_outputs)


def neighbour_exchange_bidir_with_grad(left_rank, right_rank, tensor_to_left, tensor_to_right, group=None):
    return NeighbourExchangeBidir.apply(left_rank, right_rank, group, tensor_to_left, tensor_to_right)

class SigLipLoss(nn.Module):
    """ Sigmoid Loss for Language Image Pre-Training (SigLIP) - https://arxiv.org/abs/2303.15343

    @article{zhai2023sigmoid,
      title={Sigmoid loss for language image pre-training},
      author={Zhai, Xiaohua and Mustafa, Basil and Kolesnikov, Alexander and Beyer, Lucas},
      journal={arXiv preprint arXiv:2303.15343},
      year={2023}
    }
    """
    def __init__(
            self,
            cache_labels=False,
            rank=0,
            world_size=1,
            bidir=True,
            use_horovod=False,
    ):
        super().__init__()
        self.cache_labels = cache_labels
        self.rank = rank
        self.world_size = world_size
        assert not use_horovod  # FIXME need to look at hvd ops for ring transfers
        self.use_horovod = use_horovod
        self.bidir = bidir

        # cache state FIXME cache not currently used, worthwhile?
        self.prev_num_logits = 0
        self.labels = {}

    def get_ground_truth(self, device, dtype, num_logits, negative_only=False) -> torch.Tensor:
        labels = -torch.ones((num_logits, num_logits), device=device, dtype=dtype)
        if not negative_only:
            labels = 2 * torch.eye(num_logits, device=device, dtype=dtype) + labels
        return labels

    def get_logits(self, image_features, text_features, logit_scale, logit_bias=None):
        logits = logit_scale * image_features @ text_features.T
        if logit_bias is not None:
            logits += logit_bias
        return logits

    def _loss(self, image_features, text_features, logit_scale, logit_bias=None, negative_only=False):
        logits = self.get_logits(image_features, text_features, logit_scale, logit_bias)
        labels = self.get_ground_truth(
            image_features.device,
            image_features.dtype,
            image_features.shape[0],
            negative_only=negative_only,
        )
        loss = -F.logsigmoid(labels * logits).sum() / image_features.shape[0]
        return loss

    def forward(self, image_features, text_features, logit_scale, logit_bias, output_dict=False):
        loss = self._loss(image_features, text_features, logit_scale, logit_bias)

        if self.world_size > 1:
            # exchange text features w/ neighbour world_size - 1 times
            right_rank = (self.rank + 1) % self.world_size
            left_rank = (self.rank - 1 + self.world_size) % self.world_size
            if self.bidir:
                text_features_to_right = text_features_to_left = text_features
                num_bidir, remainder = divmod(self.world_size - 1, 2)
                for i in range(num_bidir):
                    text_features_recv = neighbour_exchange_bidir_with_grad(
                        left_rank,
                        right_rank,
                        text_features_to_left,
                        text_features_to_right,
                    )

                    for f in text_features_recv:
                        loss += self._loss(
                            image_features,
                            f,
                            logit_scale,
                            logit_bias,
                            negative_only=True,
                        )
                    text_features_to_left, text_features_to_right = text_features_recv

                if remainder:
                    text_features_recv = neighbour_exchange_with_grad(
                        left_rank, right_rank, text_features_to_right)

                    loss += self._loss(
                        image_features,
                        text_features_recv,
                        logit_scale,
                        logit_bias,
                        negative_only=True,
                    )
            else:
                text_features_to_right = text_features
                for i in range(self.world_size - 1):
                    text_features_from_left = neighbour_exchange_with_grad(
                        left_rank, right_rank, text_features_to_right)

                    loss += self._loss(
                        image_features,
                        text_features_from_left,
                        logit_scale,
                        logit_bias,
                        negative_only=True,
                    )
                    text_features_to_right = text_features_from_left

        return {"contrastive_loss": loss} if output_dict else loss


class SpuriousKLReg(nn.Module):
    def forward(self,
                image_features,
                text_features,
                updated_image_features,
                updated_text_features,
                logit_scale,
                updated_logit_scale
                ):
         
        logits_per_image = logit_scale * image_features @ text_features.T
        # logits_per_image_updated = updated_logit_scale * updated_image_features @ updated_text_features.T
        logits_per_image_updated = logit_scale * updated_image_features @ updated_text_features.T

    

        reg = F.kl_div(F.log_softmax(logits_per_image_updated, -1), logits_per_image.softmax(-1), reduction='batchmean')
        # reg = F.kl_div(F.log_softmax(logits_per_image_updated, -1), logits_per_image.softmax(-1), reduction='batchmean') + F.kl_div(F.log_softmax(logits_per_text_updated, -1), logits_per_text.softmax(-1), reduction='batchmean')


        return reg

class StarReg(nn.Module):
    def forward(self,
                image_features_t,
                text_features_spurious_t,
                image_features_s,
                text_features_spurious_s,
                logit_scale,
                label):

        N = image_features_t.size(0)
        L = (label.view(-1, 1) == label.view(1, -1))
        with torch.no_grad():
            logits_per_image_t = logit_scale * image_features_t @ text_features_spurious_t.T
            logits_per_image_t[L] = float('-1e8')
            
        logits_per_image_s = logit_scale * image_features_s @ text_features_spurious_s.T
        logits_per_image_s[L] = float('-1e8')
        
        reg = F.kl_div(F.log_softmax(logits_per_image_s, dim=-1), logits_per_image_t.softmax(-1), reduction='batchmean')
        
        return reg
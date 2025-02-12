from __future__ import division, print_function, absolute_import
from torchreid import metrics
from torchreid.losses import TripletLoss, CrossEntropyLoss, DiversityLoss, BEDLoss

from ..engine import Engine


class ImageDiversityEngine(Engine):
    r"""Diversity-loss engine for image-reid.

    Args:
        datamanager (DataManager): an instance of ``torchreid.data.ImageDataManager``
            or ``torchreid.data.VideoDataManager``.
        model (nn.Module): model instance.
        optimizer (Optimizer): an Optimizer.
        margin (float, optional): margin for triplet loss. Default is 0.3.
        weight_t (float, optional): weight for triplet loss. Default is 1.
        weight_x (float, optional): weight for softmax loss. Default is 1.
        scheduler (LRScheduler, optional): if None, no learning rate decay will be performed.
        use_gpu (bool, optional): use gpu. Default is True.
        label_smooth (bool, optional): use label smoothing regularizer. Default is True.

    Examples::

        import torchreid
        datamanager = torchreid.data.ImageDataManager(
            root='path/to/reid-data',
            sources='market1501',
            height=256,
            width=128,
            combineall=False,
            batch_size=32,
            num_instances=4,
            train_sampler='RandomIdentitySampler' # this is important
        )
        model = torchreid.models.build_model(
            name='resnet50',
            num_classes=datamanager.num_train_pids,
            loss='diverse'
        )
        model = model.cuda()
        optimizer = torchreid.optim.build_optimizer(
            model, optim='adam', lr=0.0003
        )
        scheduler = torchreid.optim.build_lr_scheduler(
            optimizer,
            lr_scheduler='single_step',
            stepsize=20
        )
        engine = torchreid.engine.ImageTripletEngine(
            datamanager, model, optimizer, margin=0.3,
            weight_t=0.7, weight_x=1, scheduler=scheduler
        )
        engine.run(
            max_epoch=60,
            save_dir='log/resnet50-triplet-market1501',
            print_freq=10
        )
    """

    def __init__(
        self,
        datamanager,
        model,
        optimizer,
        templates=8,
        margin=0.3,
        alpha=0.3,
        weight_t=1,
        weight_x=1,
        weight_d=1,
        weight_b=0,
        scheduler=None,
        use_gpu=True,
        label_smooth=True,
        fused=False,
    ):
        super(ImageDiversityEngine, self).__init__(datamanager, use_gpu)

        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.register_model('model', model, optimizer, scheduler)

        assert weight_t >= 0 and weight_x >= 0 and weight_d >= 0
        assert weight_t + weight_x + weight_d > 0
        self.weight_t = weight_t
        self.weight_x = weight_x
        self.weight_d = weight_d
        self.weight_b = weight_b
        self.templates = templates

        self.criterion_t = TripletLoss(margin=margin)
        self.criterion_b = BEDLoss(alpha=alpha)
        self.criterion_x = CrossEntropyLoss(
            num_classes=self.datamanager.num_train_pids,
            use_gpu=self.use_gpu,
            label_smooth=label_smooth
        )
        self.criterion_d = DiversityLoss(n_templates=templates)

    def forward_backward(self, data):
        imgs, pids = self.parse_data_for_train(data)

        if self.use_gpu:
            imgs = imgs.cuda()
            pids = pids.cuda()

        outputs, features = self.model(imgs)

        loss = 0
        loss_summary = {}

        if self.weight_t > 0:
            loss_t = self.compute_loss(self.criterion_t, features, pids)
            loss += self.weight_t * loss_t
            loss_summary['loss_t'] = loss_t.item()

        if self.weight_d > 0 and isinstance(features, (tuple, list)):
            if self.fused:
                feature_list = features[0].reshape(features[0].shape[0], -1, self.model.embed_dim)
                feature_list = [feature_list[:, i, :] for i in range(feature_list.shape[1])]
                loss_d = self.compute_loss(self.criterion_d,
                                           features[1:self.templates + 1],
                                           pids)
            else:
                loss_d = self.compute_loss(self.criterion_d,
                                           features[1:self.templates + 1],
                                           pids)
            loss += self.weight_d * loss_d
            loss_summary['loss_d'] = loss_d.item()

        if self.weight_x > 0:
            loss_x = self.compute_loss(self.criterion_x, outputs, pids)
            loss += self.weight_x * loss_x
            loss_summary['loss_x'] = loss_x.item()
            loss_summary['acc_global'] = metrics.accuracy(outputs, pids)[0].item()
            if not self.fused:
                if isinstance(outputs, (tuple, list)):
                    for part_output in outputs[1:self.templates + 1]:
                        if 'acc_part' not in loss_summary.keys():
                            loss_summary['acc_part'] = metrics.accuracy(part_output, pids)[0].item()
                        else:
                            loss_summary['acc_part'] += metrics.accuracy(part_output, pids)[0].item()
                    loss_summary['acc_part'] /= len(outputs[1:])
                    if len(outputs) > self.templates + 1:
                        loss_summary['acc_graph'] = metrics.accuracy(outputs[-1], pids)[0].item()
        assert loss_summary

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss_summary

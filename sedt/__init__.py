from utilities.utils import to_cuda_if_available
from .spsedt import SPSEDT
from .sedt import SEDT, SetCriterion, PostProcess
from .backbone import build_backbone
from .transformer import build_transformer, TransformerDecoder, TransformerDecoderLayer
from .matcher import build_matcher

def build_model(args):
    if args.self_sup:
        num_classes = 1
    else:
        num_classes = args.num_classes

    backbone = build_backbone(args)
    transformer = build_transformer(args)
    if args.self_sup:
        model = SPSEDT(
            backbone,
            transformer,
            num_classes=num_classes,
            num_queries=args.num_queries,
            aux_loss=args.aux_loss,
            feature_recon=args.feature_recon,
            query_shuffle=args.query_shuffle,
            num_patches=args.num_patches
        )
    else:
        model = SEDT(
            backbone,
            transformer,
            num_classes=num_classes,
            num_queries=args.num_queries,
            aux_loss=args.aux_loss,
            dec_at=args.dec_at,
            pooling=args.pooling
        )
    matcher = build_matcher(args)
    weight_dict = {'loss_ce': args.ce_loss_coef, 'loss_bbox': args.bbox_loss_coef, 'loss_giou': args.giou_loss_coef}
    losses = ['labels', 'boxes', 'cardinality']
    if not args.self_sup:
        if args.dec_at:
            weight_dict['loss_weak'] = args.weak_loss_coef
            losses += ['weak']
        if args.pooling:
            weight_dict['loss_weak_p'] = args.weak_loss_p_coef
    else:
        if args.feature_recon:
            losses += ['feature']
            weight_dict['loss_feature'] = 1

    # TODO this is a hack
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    criterion = SetCriterion(num_classes, matcher=matcher, weight_dict=weight_dict,
                             eos_coef=args.eos_coef, losses=losses)
    criterion = to_cuda_if_available(criterion)
    postprocessors = {'bbox': PostProcess()}

    return model, criterion, postprocessors





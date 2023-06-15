from statistics import mode
from scene_graph_benchmark.scene_parser import SceneParser
from maskrcnn_benchmark.solver import make_optimizer
from maskrcnn_benchmark.solver import make_lr_scheduler
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
from maskrcnn_benchmark.utils.comm import synchronize, get_rank
def construct_module(cfg):
        model = SceneParser(cfg)

        model.cuda()
        optimizer = make_optimizer(cfg, model)
        scheduler = make_lr_scheduler(cfg, optimizer)
        output_dir = cfg.OUTPUT_DIR
        save_to_disk = get_rank() == 0
        checkpointer = DetectronCheckpointer(
            cfg, model, optimizer, scheduler, output_dir, save_to_disk
        )
        extra_checkpoint_data = checkpointer.load(cfg.MODEL.WEIGHT)

        return model


def load_models(cfg,flag):
    # print('load_models:', flag)
    model = construct_module(cfg)
    model.name = flag
    return model

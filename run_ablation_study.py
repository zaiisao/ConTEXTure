import sys
import tempfile
import pyrallis

from src.training.trainer import TEXTure
from src.configs.train_config import TrainConfig

sys.path.append("./src/zero123/zero123")
sys.path.append("./src/zero123/ControlNet")

for crossattn in range(1, 10, 2):
    for concat in range(1, 10, 2):
        for control in range(1, 10, 2):
            with tempfile.NamedTemporaryFile(mode='w+') as fp:
                fp.write(f"""log:
  exp_name: spiderman_{crossattn}_{concat}_{control}
guide:
  text: "A photo of Spiderman, {'{}'} view"
  append_direction: True
  shape_path: shapes/human.obj
  guidance_scale: 10
  second_model_type: "control_zero123"

  individual_control_of_conditions: True
  guidance_scale_crossattn: {crossattn}
  guidance_scale_concat: {concat}
  guidance_scale_control: {control}

  guess_mode: True
  use_inpainting: False""")
                fp.flush()

                @pyrallis.wrap(config_path=fp.name)
                def main(cfg: TrainConfig):
                    trainer = TEXTure(cfg)
                    if cfg.log.eval_only:
                        trainer.full_eval()
                    else:
                        trainer.paint()

                main()

                # os.system(f'python -m scripts.run_texture --config_path={fp.name}')
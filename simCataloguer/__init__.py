from typing import Optional, Union
import gpt_2_simple as gpt2
import sys
from pathlib import Path, PurePath
import tensorflow as tf
import pixray_module

def get_rel_path(directory: Union[str, Path], default_dir: Path) -> Path:
    if isinstance(directory, Path):
        return directory
    if PurePath(directory).is_absolute():
        return Path(directory)
    else:
        return default_dir.parent / Path(directory)

def generate(
    session, 
    run_name: str, 
    length: int = 500, 
    temperature: float = 0.8, 
    n_samples: int = 1, 
    batch_size: int = 1
) -> list[str]:
    return gpt2.generate(
        session, 
        run_name=run_name, 
        length=length, 
        temperature=temperature,
        nsamples=n_samples, 
        batch_size=batch_size,
        return_as_list = True
    )


class Writer:
    def __init__(
        self, 
        model_dir: Union[str, Path] = 'current_models', 
        model_name='124M'
    ) -> None:
        self.default_dir = Path.cwd() / sys.argv[0]
        self.model_name = model_name
        self.model_path = get_rel_path(model_dir, self.default_dir)
        has_downloaded_path = self.model_path/'model_has_downloaded.lock'
        if not has_downloaded_path.exists():
            gpt2.download_gpt2(model_dir = str(self.model_path), model_name=model_name)
            with open(has_downloaded_path, 'w+'):
                pass

        self._sess = None
        self.model_was_trained: bool = False

    @property
    def sess(self):
        if self._sess is None:
            self._sess = gpt2.start_tf_sess()
        return self._sess

    def _train_model(
            self, 
            file_name: Union[str, Path], 
            run_name: str,
            checkpoint_path: Path,  
        ):
        train_path = get_rel_path(file_name, self.default_dir)
        
        tf.compat.v1.reset_default_graph()
        gpt2.finetune(self.sess,
            dataset=str(train_path),
            model_name=self.model_name,
            steps=100,
            restore_from='fresh',
            run_name=run_name,
            print_every=10,
            sample_every=20,
            save_every=100,
            checkpoint_dir = str(checkpoint_path),
            model_dir = str(self.model_path)
        )
        self.model_was_trained = True

    def train_model(
        self, 
        file_name: Union[str, Path], 
        run_name: str,
        checkpoint_dir: Union[str, Path] = 'checkpoint',  
        always_retrain: bool = False
    ):
        if not isinstance(checkpoint_dir, Path):
            checkpoint_dir = Path(checkpoint_dir)
            checkpoint_path = get_rel_path(checkpoint_dir, self.default_dir)
        else:
            checkpoint_path = checkpoint_dir

        if always_retrain or not (Path(checkpoint_dir) / run_name).exists():
            self._train_model(file_name, run_name, checkpoint_path)
        else:
            gpt2.load_gpt2(self.sess, run_name = run_name, checkpoint_dir= str(checkpoint_path)  )

    def make_pixray(
        self,
        prompt: str,
        output: Union[str, Path],
        quality: str = "better",
        num_cuts: int = 10
    ):
        if not isinstance(output, Path):
            output = Path(output)
        pixray_module.run(
            prompt[0:70],
            "vdiff",
            quality=quality,
            custom_loss="edge, symmetry",
            edge_color="grey",
            num_cuts=num_cuts,
            outdir=str(output.absolute()),
            make_video = False,
            save_intermediates = False
        )

    def __call__(
        self,
        input_file: Union[str, Path], 
        run_name: str, 
        output: Union[str, Path],
        choose_or_refuse: bool = True,
        checkpoint_dir: Union[str, Path] = 'checkpoint',
        text_length: int = 500, 
        text_temperature: float = 0.8, 
        always_retrain: bool = False,
        pixray_quality: str = "better",
        pixray_num_cuts: int = 10
    ):
        if not self.model_was_trained:
            self.train_model(
                file_name=input_file, 
                run_name=run_name, 
                checkpoint_dir=checkpoint_dir,
                always_retrain = always_retrain
            )
        if choose_or_refuse:
            output_text: Optional[str] = None
            while output_text is None:
                generated_texts = generate(
                    self.sess, run_name=run_name, length=text_length, temperature=text_temperature
                )
                print()
                print(generated_texts[0])
                print()
                while True:
                    response = input('Is this text acceptable? (y/n):')
                    if response.lower() == 'y':
                        output_text = generated_texts[0]
                        break
                    elif response.lower() == 'n':
                        break
                    else:
                        print('invalid_response')

        else:
            generated_texts = generate(
                self.sess, run_name=run_name, length=text_length, temperature=text_temperature
            )
            output_text = generated_texts[0]
        self.make_pixray(output_text, output, pixray_quality, pixray_num_cuts)


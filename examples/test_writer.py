from make_writer import Writer
import pixray_module
writer = Writer()
writer.train_model('obama_sotu.txt', 'simObama')
print(writer.generate(run_name='simObama'))

from pathlib import Path
path = Path("outputs/hairout")

pixray_module.run(
    "pandas made of shiny metal",
    "vdiff",
    quality="draft",
    custom_loss="edge, symmetry",
    edge_color="grey",
    num_cuts=2,
    outdir=str(path.absolute()),
    make_video = False,
    save_intermediates = False
)

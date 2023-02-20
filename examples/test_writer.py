from simCataloguer import Writer
import pixray_module
from pathlib import Path
writer = Writer()
writer.train_model('obama_sotu.txt', 'simObama')
print(writer.generate(run_name='simObama'))


path = Path("outputs/hairout")

pixray_module.run(
    "pandas made of shiny metal",
    "vdiff",
    quality="better",
    custom_loss="edge, symmetry",
    edge_color="grey",
    num_cuts=10,
    outdir=str(path.absolute()),
    make_video = False,
    save_intermediates = False
)

"""Execute every notebook cell and preserve real outputs."""
from pathlib import Path
import os
import nbformat
from nbclient import NotebookClient
ROOT=Path(__file__).resolve().parents[1]
runtime=ROOT/'outputs'/'jupyter_runtime'
runtime.mkdir(parents=True,exist_ok=True)
os.environ.setdefault('JUPYTER_RUNTIME_DIR',str(runtime))
os.environ.setdefault('IPYTHONDIR',str(ROOT/'outputs'/'ipython'))
path=ROOT/'assignment3.ipynb'
notebook=nbformat.read(path,as_version=4)
NotebookClient(notebook,timeout=1200,kernel_name='assignment3',
               resources={'metadata':{'path':str(ROOT)}}).execute()
errors=[out for cell in notebook.cells if cell.cell_type=='code'
        for out in cell.get('outputs',[]) if out.output_type=='error']
assert not errors,errors
nbformat.validate(notebook)
nbformat.write(notebook,path)
print(f'Executed {sum(c.cell_type=="code" for c in notebook.cells)} code cells with zero errors; outputs saved.')

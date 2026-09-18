"""Execute every notebook cell and preserve real outputs, never fabricated cells."""
from pathlib import Path
import nbformat
from nbclient import NotebookClient
ROOT=Path(__file__).resolve().parents[1]
path=ROOT/'assignment3.ipynb'
notebook=nbformat.read(path,as_version=4)
NotebookClient(notebook,timeout=1200,kernel_name='assignment3',resources={'metadata':{'path':str(ROOT)}}).execute()
errors=[out for c in notebook.cells if c.cell_type=='code' for out in c.get('outputs',[]) if out.output_type=='error']
assert not errors,errors
nbformat.validate(notebook);nbformat.write(notebook,path)
print(f'Executed {sum(c.cell_type=="code" for c in notebook.cells)} code cells with zero errors; outputs saved.')

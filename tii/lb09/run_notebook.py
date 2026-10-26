"""Execute the existing notebook using this local virtual environment."""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)
work = ROOT / '.work'
work.mkdir(exist_ok=True)
for key in ('TEMP', 'TMP', 'JUPYTER_RUNTIME_DIR'):
    os.environ[key] = str(work)
os.environ['IPYTHONDIR'] = str(work / 'ipython')
ENV = Path(sys.prefix)
os.environ['JUPYTER_PATH'] = str(ENV / 'share/jupyter')
import nbformat
from nbclient import NotebookClient
from ipykernel.kernelspec import install

install(prefix=str(ENV), kernel_name='lb9', display_name='Python (lb9 .venv)')
path = ROOT / '09_student.ipynb'
nb = nbformat.read(path, as_version=4)
nb.metadata.kernelspec = {'name': 'lb9', 'display_name': 'Python (lb9 .venv)', 'language': 'python'}
NotebookClient(nb, timeout=180, startup_timeout=45, kernel_name='lb9',
               resources={'metadata': {'path': str(ROOT)}}).execute()
nbformat.write(nb, path)
print('All notebook cells executed successfully with', sys.executable)

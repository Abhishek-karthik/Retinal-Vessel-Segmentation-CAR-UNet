import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys

# Pre-import torch in main process to ensure DLLs are loaded
import torch
import nbformat
import time

print(f"PyTorch pre-loaded: {torch.__version__} | CUDA: {torch.cuda.is_available()}")

notebook_path = sys.argv[1] if len(sys.argv) > 1 else "Review2_Complete_CAR_UNet_Pipeline.ipynb"
with open(notebook_path, "r", encoding="utf-8") as f:
    nb = json_nb = nbformat.read(f, as_version=4)

print(f"Executing {notebook_path} cell by cell in local Python environment...")

# Execute each code cell sequentially and capture rich outputs
import io
from contextlib import redirect_stdout, redirect_stderr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import base64

globs = {
    "__name__": "__main__",
    "__file__": os.path.abspath(notebook_path)
}

start_time = time.time()
executed_cells = 0
failed_cells = []

for idx, cell in enumerate(nb.cells):
    if cell.cell_type == "code":
        executed_cells += 1
        code = cell.source
        print(f"  >>> Executing Cell {idx+1} ...")
        
        stdout_buf = io.StringIO()
        stderr_buf = io.StringIO()
        cell_outputs = []
        
        plt.close('all')
        
        try:
            with redirect_stdout(stdout_buf), redirect_stderr(stderr_buf):
                exec(code, globs)
            
            # Check for stdout
            out_text = stdout_buf.getvalue()
            if out_text:
                cell_outputs.append(nbformat.v4.new_output(
                    output_type="stream",
                    name="stdout",
                    text=out_text
                ))
            
            # Check for matplotlib figures
            fig_nums = plt.get_fignums()
            for fig_num in fig_nums:
                fig = plt.figure(fig_num)
                buf = io.BytesIO()
                fig.savefig(buf, format="png", bbox_inches="tight", dpi=150)
                buf.seek(0)
                img_base64 = base64.b64encode(buf.read()).decode("utf-8")
                cell_outputs.append(nbformat.v4.new_output(
                    output_type="display_data",
                    data={"image/png": img_base64, "text/plain": "<Figure size ...>"}
                ))
                plt.close(fig)
                
            cell.outputs = cell_outputs
            cell.execution_count = executed_cells
            
        except Exception as e:
            print(f"Error in cell {idx+1}: {e}")
            failed_cells.append(idx)
            import traceback
            tb = traceback.format_exc()
            cell_outputs.append(nbformat.v4.new_output(
                output_type="error",
                ename=type(e).__name__,
                evalue=str(e),
                traceback=tb.splitlines()
            ))
            cell.outputs = cell_outputs
            cell.execution_count = executed_cells

total_duration = time.time() - start_time
if failed_cells:
    print(f"\n{len(failed_cells)} of {executed_cells} code cells FAILED (notebook cell indices {failed_cells}) in {total_duration:.2f} seconds.")
else:
    print(f"\nAll {executed_cells} code cells executed successfully in {total_duration:.2f} seconds!")

with open(notebook_path, "w", encoding="utf-8") as f:
    nbformat.write(nb, f)

print(f"Saved executed notebook to: {notebook_path}")
if failed_cells:
    sys.exit(1)

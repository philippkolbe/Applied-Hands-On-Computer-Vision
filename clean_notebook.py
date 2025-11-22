import nbformat

path = "MNIST_Curated_LeNet_with_idk_labels.ipynb"

nb = nbformat.read(path, as_version=4)

# Remove widget state entirely
if "widgets" in nb["metadata"]:
    del nb["metadata"]["widgets"]

# Remove cell widget metadata too
for cell in nb["cells"]:
    if "metadata" in cell and "widgets" in cell["metadata"]:
        del cell["metadata"]["widgets"]

nbformat.write(nb, path)

print("Cleaned notebook saved!")
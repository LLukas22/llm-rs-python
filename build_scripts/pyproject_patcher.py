import toml
import os
import sys

if __name__ == "__main__":

    file = "pyproject.toml"
    if len(sys.argv) > 1:
        file = sys.argv[1]
        
    name = os.environ.get("PYPROJECT_NAME","llm-rs")

    try:
        data = toml.load(open(file,"r",encoding="utf-8"))
        data["project"]["name"] = name
        toml.dump(data,open("pyproject.toml.new","w",encoding="utf-8"))
        #Only replace if no error occured
        os.remove("pyproject.toml")
        os.rename("pyproject.toml.new","pyproject.toml")
        print(f"Changed projectname of file {file} to {name}!")
    except Exception as e:
        print(f"Could not change projectname of file {file} to {name}!")
        print(e)



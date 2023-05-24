from huggingface_hub import HfApi, CommitOperationAdd, CommitOperationDelete
from huggingface_hub import create_repo,metadata_update,cached_assets_path
from typing import List, Optional, Union
import json
import os
import pathlib
import logging 

class Repository():
    def __init__(self,name:str,user_or_organization:Optional[str]=None,token:Optional[str]=None,private:bool=False) -> None:
        
        self.name = f"{user_or_organization}/{name}" if user_or_organization else name
        self.api = HfApi(token=token)

        try:
            self.url = str(create_repo(self.name,token=token,private=private))
            self.name = self.url.replace(f"{self.api.endpoint}/","")
            # Uplaod the config file
            config_path = cached_assets_path("llm-rs",namespace=self.name) / "config.json"
            with open(config_path,"w") as f:
                f.write(json.dumps({"repo_type":"GGML"}))
            self.api.create_commit(
                self.name,
                operations=[CommitOperationAdd(path_in_repo="config.json", path_or_fileobj=config_path)],
                commit_message="Auto initialized repo via llm-rs",)
            
            metadata={}
            metadata["pipeline_tag"]= "text-generation"
            metadata["tags"] = ["llm-rs","ggml"]
            metadata_update(self.name, metadata, token=token)
            
        except Exception as e:
            self.url = create_repo(self.name,token=token,private=private,exist_ok=True)
            self.name = self.url.replace(f"{self.api.endpoint}/","")

    def upload(self,model_file:Union[str,os.PathLike],delete_old:bool=True):
        # search for the metadata file
        path = pathlib.Path(model_file)
        metadata_path = path.with_suffix(".meta")

        if not path.exists():
            raise FileNotFoundError(f"Could not find model file {model_file}")
        
        if delete_old:
            to_delete = [path.name]
            if metadata_path.exists():
                to_delete.append(metadata_path.name)

            for n in to_delete:
                try:
                    self.api.create_commit(
                        self.name,
                        operations=[CommitOperationDelete(path_in_repo=n,is_folder=False)],
                        commit_message=f"Delete old file: '{n}'",
                        )
                except Exception as e:
                    logging.error(f"Could not delete old file {n} from repo {self.name}")
            

        upload_operations = [CommitOperationAdd(path_in_repo=path.name, path_or_fileobj=path)] 
        if metadata_path.exists():
            upload_operations.append(CommitOperationAdd(path_in_repo=metadata_path.name, path_or_fileobj=metadata_path))
        
        self.api.create_commit(
            self.name,
            upload_operations,
            commit_message=f"Upload new model file: '{path.name}'",)


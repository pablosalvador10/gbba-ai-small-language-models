from typing import Optional
from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient
from azure.ai.ml.entities import Environment, Data
from azure.ai.ml.entities import AmlCompute
from azure.ai.ml.constants import AssetTypes
from azure.core.exceptions import ResourceNotFoundError, ResourceExistsError

from utils.ml_logging import get_logger

class AMLManager:
    """
    A helper class for managing Azure Machine Learning resources such as environments and data assets.

    Attributes:
        subscription_id (str): Azure subscription ID.
        resource_group (str): Azure resource group name.
        workspace_name (str): Azure Machine Learning workspace name.
        ml_client (MLClient): Instance of Azure Machine Learning client.
    """

    def __init__(self, subscription_id: str, resource_group: str, workspace_name: str):
        """
        Initializes the AMLHelper with Azure subscription, resource group, and workspace details.

        Parameters:
            subscription_id (str): Azure subscription ID.
            resource_group (str): Azure resource group name.
            workspace_name (str): Azure Machine Learning workspace name.
        """
        self.logger = get_logger()
        self.subscription_id = subscription_id
        self.resource_group = resource_group
        self.workspace_name = workspace_name
        self.ml_client = self._initialize_ml_client()

    def _initialize_ml_client(self) -> MLClient:
        """
        Initializes and returns the MLClient for Azure Machine Learning operations.

        Returns:
            MLClient: Initialized MLClient object.
        """
        credential = DefaultAzureCredential()
        return MLClient(credential, self.subscription_id, self.resource_group, self.workspace_name)

    def get_or_create_environment_asset(self, env_name: str, conda_yml: str = "cloud/conda.yml", update: bool = False) -> Environment:
        """
        Gets or creates an Azure Machine Learning environment asset.

        Parameters:
            env_name (str): The name of the environment.
            conda_yml (str): Path to the Conda environment YAML file.
            update (bool): Whether to update the environment if it exists.

        Returns:
            Environment: The retrieved or created environment asset.
        """
        try:
            latest_env_version = max([int(e.version) for e in self.ml_client.environments.list(name=env_name)], default=0)
            if update:
                raise ResourceExistsError('Found Environment asset, but will update the Environment.')
            else:
                env_asset = self.ml_client.environments.get(name=env_name, version=latest_env_version)
                self.logger.info(f"Found Environment asset: {env_name}. Will not create again.")
        except (ResourceNotFoundError, ResourceExistsError) as e:
            self.logger.exception(f"Exception: {e}")
            env_docker_image = Environment(
                image="mcr.microsoft.com/azureml/curated/acft-hf-nlp-gpu:latest",
                conda_file=conda_yml,
                name=env_name,
                description="Environment created for llm fine-tuning."
            )
            env_asset = self.ml_client.environments.create_or_update(env_docker_image)
            self.logger.info(f"Created Environment asset: {env_name}.")

        return env_asset

    def get_or_create_data_asset(self, data_name: str, data_local_dir: str, update: bool = False) -> Data:
        """
        Gets or creates an Azure Machine Learning data asset.

        Parameters:
            data_name (str): The name of the data asset.
            data_local_dir (str): Local directory path of the data.
            update (bool): Whether to update the data asset if it exists.

        Returns:
            Data: The retrieved or created data asset.
        """
        try:
            latest_data_version = max([int(d.version) for d in self.ml_client.data.list(name=data_name)], default=0)
            if update:
                raise ResourceExistsError('Found Data asset, but will update the Data.')
            else:
                data_asset = self.ml_client.data.get(name=data_name, version=latest_data_version)
                self.logger.info(f"Found Data asset: {data_name}. Will not create again.")
        except (ResourceNotFoundError, ResourceExistsError) as e:
            data = Data(
                path=data_local_dir,
                type=AssetTypes.URI_FOLDER,
                description=f"{data_name} for fine tuning",
                tags={"FineTuningType": "Instruction", "Language": "En"},
                name=data_name
            )
            data_asset = self.ml_client.data.create_or_update(data)
            self.logger.info(f"Created Data asset: {data_name}.")

        return data_asset
    
    def create_or_reuse_compute_cluster(self, cluster_name, cluster_size, tier="lowpriority", max_instances=1):
        """
        Attempts to retrieve an existing compute cluster by name. If it does not exist, it creates a new one.

        Parameters:
        - cluster_name: The name of the compute cluster.
        - cluster_size: The size of the compute cluster to create if it doesn't exist.
        - tier: The tier of the compute cluster, default is "lowpriority".
        - max_instances: The maximum number of instances for the compute cluster, default is 1.

        Returns:
        A compute cluster object.
        """
        try:
            # Try to get the existing compute cluster
            compute = self.ml_client.compute.get(cluster_name)
            self.logger.info("The compute cluster already exists! Reusing it for the current run.")
        except Exception as ex:
            self.logger.info(f"Compute cluster '{cluster_name}' does not exist. Creating a new one with size '{cluster_size}', tier '{tier}', and max instances '{max_instances}'.")
            try:
                # Create a new compute cluster
                compute = AmlCompute(
                    name=cluster_name,
                    size=cluster_size,
                    tier=tier,
                    max_instances=max_instances,
                )
                self.ml_client.compute.begin_create_or_update(compute).wait()
                self.logger.info(f"Compute cluster '{cluster_name}' created successfully.")
            except Exception as e:
                self.logger.error(f"Failed to create compute cluster '{cluster_name}'. Error: {e}")
                raise
        return compute
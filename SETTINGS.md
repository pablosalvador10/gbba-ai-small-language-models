## Table of Contents

- [Setting Up Azure AI Services](#setting-up-azure-ai-services)
- [Configuration Environment Variables](#configuration-environment-variables)
- [Notebooks Setup](#notebooks-setup)
  - [Setting Up Conda Environment and Configuring VSCode for Jupyter Notebooks](#setting-up-conda-environment-and-configuring-vscode-for-jupyter-notebooks)

## Setting Up Azure AI Services

- **Azure Open AI Services**: To effectively vectorize data, we leverage the `ada` model within Azure OpenAI Services. This model, part of the suite of large language and generative AI models, is specifically designed for tasks that require nuanced understanding and processing of complex data. In addition to `ada`, we also use GPT-4 Turbo for chat completions, providing a more interactive and dynamic user experience. [start here](https://techcommunity.microsoft.com/t5/ai-azure-ai-services-blog/azure-openai-service-launches-gpt-4-turbo-and-gpt-3-5-turbo-1106/ba-p/3985962)

## Configuration Environment Variables

Before running this notebook, you must configure certain environment variables. We will now use environment variables to store our configuration. This is a more secure practice as it prevents sensitive data from being accidentally committed and pushed to version control systems.

Create a `.env` file in your project root (use the provided `.env.sample` as a template) and add the following variables:

```env
# Azure Open AI Completion Configuration
AZURE_OPENAI_KEY="your_azure_openai_key"  # Replace with your Azure OpenAI Key
AZURE_AOAI_CHAT_MODEL_NAME_DEPLOYMENT_ID="your_chat_model_name_deploiment_id"  # Optional
AZURE_AOAI_COMPLETION_MODEL_DEPLOYMENT_ID="your_completion_model_deployment_id"  # Optional
AZURE_AOAI_EMBEDDING_DEPLOYMENT_ID="your_azure_aoai_embedding_deployment_id"  # Optional
AZURE_AOAI_VISION_MODEL_DEPLOYMENT_ID=your_vision_model_deployment_id
AZURE_AOAI_DALLE_MODEL_DEPLOYMENT_ID=your_dalle_model_deployment_id
AZURE_OPENAI_API_ENDPOINT="your_azure_openai_api_endpoint"  # Replace with your Azure OpenAI API Endpoint
AZURE_OPENAI_API_VERSION="your_azure_openai_api_version"  # Replace with your Azure OpenAI API Version
```

- `AZURE_OPENAI_KEY`: This is your Azure OpenAI Key. You can find it in the "Keys" section of your OpenAI Service in the Azure portal.
- `AZURE_AOAI_CHAT_MODEL_NAME`: This is the name of your Chat Model. You define this when you create a model in your OpenAI Service.
- `AZURE_AOAI_COMPLETION_MODEL_DEPLOYMENT_ID`: This is the Deployment ID of your Completion Model. You can find it in the "Deployments" section of your OpenAI Service in the Azure portal. This is optional and can be set to 'none' if you do not want to use the model.
- `AZURE_AOAI_EMBEDDING_DEPLOYMENT_ID`: This is the Deployment ID of your Azure AOAI Embedding. You can find it in the "Deployments" section of your OpenAI Service in the Azure portal. This is optional and can be set to 'none' if you do not want to use the model.
- `AZURE_AOAI_VISION_MODEL_DEPLOYMENT_ID`: your_vision_model_deployment_id
- `AZURE_AOAI_DALLE_MODEL_DEPLOYMENT_ID`: your_dalle_model_deployment_id
- `AZURE_OPENAI_API_ENDPOINT`: This is the URL of your Azure OpenAI API. You can find it in the "Overview" section of your OpenAI Service in the Azure portal.
- `AZURE_OPENAI_API_VERSION`: This is your Azure OpenAI API Version. You define this when you set up your OpenAI Service.

> 📌 **Note**
> Remember not to commit the .env file to your version control system. Add it to your .gitignore file to prevent it from being tracked.

## Create Conda Environment from the Repository

> Instructions for Windows users:

1. **Create the Conda Environment**:
   - In your terminal or command line, navigate to the repository directory.
   - Execute the following command to create the Conda environment using the `environment.yaml` file:
     ```bash
     conda env create -f environment.yaml
     ```
   - This command creates a Conda environment as defined in `environment.yaml`.

2. **Activating the Environment**:
   - After creation, activate the new Conda environment by using:
     ```bash
     conda activate aoai-faq
     ```

> Instructions for Linux users (or Windows users with WSL or other linux setup):

1. **Use `make` to Create the Conda Environment**:
   - In your terminal or command line, navigate to the repository directory and look at the Makefile.
   - Execute the `make` command specified below to create the Conda environment using the `environment.yaml` file:
     ```bash
     make create_conda_env
     ```

2. **Activating the Environment**:
   - After creation, activate the new Conda environment by using:
     ```bash
     conda activate speech-ai-azure-services
     ```

## Configure VSCode for Jupyter Notebooks

1. **Install Required Extensions**:
   - Download and install the `Python` and `Jupyter` extensions for VSCode. These extensions provide support for running and editing Jupyter Notebooks within VSCode.

2. **Attach Kernel to VSCode**:
   - After creating the Conda environment, it should be available in the kernel selection dropdown. This dropdown is located in the top-right corner of the VSCode interface.
   - Select your newly created environment (`aoai-faq`) from the dropdown. This sets it as the kernel for running your Jupyter Notebooks.

3. **Run the Notebook**:
   - Once the kernel is attached, you can run the notebook by clicking on the "Run All" button in the top menu, or by running each cell individually.

4. **Voila! Ready to Go**:
   - Now that your environment is set up and your kernel is attached, you're ready to go! Please visit the notebooks in the repository to start exploring.

> **Note:** By following these steps, you'll establish a dedicated Conda environment for your project and configure VSCode to run Jupyter Notebooks efficiently. This environment will include all the necessary dependencies specified in your `environment.yaml` file. If you wish to add more packages or change versions, please use `pip install` in a notebook cell or in the terminal after activating the environment, and then restart the kernel. The changes should be automatically applied after the session restarts.

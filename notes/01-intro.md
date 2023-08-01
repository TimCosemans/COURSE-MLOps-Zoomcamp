# 1 Introduction
MLOps consist of three stages:  
\begin{itemize}
    \item Design: Establish the goals of the project
    \item Training: Train the model
        \subitem Experiment tracking
        \subitem Training pipeline 
        \subitem Model registry
    \item Operations: Deploy the model 
        \subitem Batch or streaming web service 
        \subitem Model monitoring
\end{itemize}

To set up the environment, create a new virtual machine in Azure: 
\begin{itemize}
    \item Ubuntu
    \item 64-bit (x86)
\end{itemize}

Then download the key and connect to the machine using the ssh-key in the terminal. 

<pre>
<code> 
ssh -i <private key path> azureuser@20.229.125.191
</pre>
</code> 

Give an alias to the config file in the .ssh folder:
<pre>
<code>

nano .ssh/config

Host <alias>
    HostName <ip address>
    User azureuser
    IdentityFile /Users/timcosemans/Library/CloudStorage/OneDrive-Cronos/mlops_key.pem
    StrictHostKeyChecking no
</code>
</pre>

Then you can connect to the VM using the alias:
<pre>
<code>
ssh <alias>
</code>
</pre>

Install python from Anaconda

<pre>
<code>
wget https://repo.anaconda.com/archive/Anaconda3-2023.03-1-Linux-x86_64.sh
bash Anaconda3-2023.03-1-Linux-x86_64.sh
</code>
</pre>

To install Docker, run the following commands:

<pre>
<code>
sudo apt update 
sudo apt install docker.io
</code>
</pre>

Then install Docker compose in a new 'soft' folder. 

<pre>
<code>
mkdir soft
cd soft
wget https://github.com/docker/compose/releases/download/v2.18.1/docker-compose-linux-x86_64 -O docker-compose
chmod +x docker-compose

#then add to path
cd .. 
nano .bashrc 

#add the following line to the end of the file
export PATH="${HOME}/soft:${PATH}"

#to then run this command 
source .bashrc
sudo groupadd docker
sudo usermod -aG docker $USER

#to try if it works 
docker run hello-world
</code>
</pre>

You can then install the Remote - SSH extension in VSCode and connect to the VM.

## Terraform
It is always good practice to setup your infrastructure using code. This way, you can easily recreate your infrastructure and you can version control it. You can either choose to do so from [scratch](https://developer.hashicorp.com/terraform/tutorials/azure-get-started/infrastructure-as-code) or [export](https://developer.hashicorp.com/terraform/tutorials/state/state-import) existing infrastructure to Terraform. Use a .tfvars file to store your credentials. You can then use the following commands to setup your infrastructure:

<pre>
<code>
terraform init
terraform plan -var-file="credentials.tfvars"
terraform apply -var-file="credentials.tfvars"
</code>
</pre>



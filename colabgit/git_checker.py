# Example:
# 
# import git_checker
# 
# git_rep_address = "https://github.com/::your_git_account_name::/::your_repository.git::"
#
# git_checker.git_pull_clone(git_rep_address)
  # первый вызов выполняет clone, если на 
  # локальной стороне нет каталога репозитория
# 
# git_checker.git_pull_clone(git_rep_address)
  # второй вызов выполняет pull
# 


import os
import subprocess

                                                                                            
def colab_git_clone(git_rep_address):
  """
  """
  
  file_name = "./git_clone.sh"
  f = open(file_name,"w+") 
    # В ряде случаев наблюдалась задержка при 
    # создании файла, поэтому сначала просто
    # создаем файл
    # .
  f.close()
  
  with open(file_name,"r+") as f_git_pull:
    sh_script = f"#!/bin/bash \n \
    git clone --depth 1 {git_rep_address} \n"
    f_git_pull.write(sh_script)
  
  subprocess.run("chmod ug+rwx ./git_clone.sh",shell=True)
    # !chmod ug+rwx ./git_clone.sh
    # colab-синтаксис 
    # os.system(f"chmod ug+rwx ./git_clone.sh")
    # не выводит в colab-консоль из коробки
  res = subprocess.run("sh ./git_clone.sh",shell=True,capture_output=True)
    # !sh ./git_clone.sh
  
  print(res.stdout.decode())
  print("\n \033[43m")
  print(res.stderr.decode())
  print("\033[0m")


def colab_git_pull(git_dir):
  """
  """
  
  file_name = "./git_pull.sh"
  f = open(file_name,"w")
    # В ряде случаев наблюдалась задержка при 
    # создании файла, поэтому сначала просто
    # создаем файл
    # .
  f.close()
  with open(file_name,"r+") as f_git_pull:
    
    sh_script = f"#!/bin/bash \n \
    cd {git_dir} \n \
    echo \n \
    ls -la \n \
    echo \n \
    echo \"\033[47mgit status\033[0m\" \n \
    echo \n \
    git status \n \
    echo \n \
    echo \"\033[47mgit pull\033[0m\" \n \
    echo \n \
    git pull \n"
    f_git_pull.write(sh_script)
  
  subprocess.run("chmod ug+rwx ./git_pull.sh",shell=True)
    # !chmod ug+rwx ./git_pull.sh
  res = subprocess.run(["sh", "./git_pull.sh"],capture_output=True)
    # !sh ./git_pull.sh
    # subprocess.run("sh ./git_pull.sh",shell=True,capture_output=True,text=True)
  print(res.stderr.decode())
  print("\033[0m")
  print(res.stdout.decode())
  print("\n \033[43m")


def git_pull_clone(git_rep_address):
  """
  """

  git_dir = f"/content/"+ git_rep_address.replace(".git","").split("/")[-1]+f"/"
    # "/content/set_for_colaboratory/"

  if os.path.isdir(git_dir):
    print("::-::-"*15+"::")
    print("\n\033[47m!git dir\033[0m\ \n")
    colab_git_pull(git_dir)
  else:
    print("::-::-"*15+"::")
    print("\n \033[47m!git clone\033[0m\ \n")
    colab_git_clone(git_rep_address)

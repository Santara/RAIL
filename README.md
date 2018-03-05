# RAIL : Risk-Averse Imitation Learning #

Codebase for [RAIL : Risk Averse Imitation Learning](https://arxiv.org/abs/1707.06658), presented at the [NIPS 2017 DRL Symposium](https://sites.google.com/view/deeprl-symposium-nips2017).   

---

## Setting up

* Set up [MuJoCo](https://www.roboti.us/index.html) on your machine. Please download `mjpro131` for compatibility with the rest of the code.
* Install [Anaconda Python 2.7](https://repo.continuum.io/archive/) and set it as the default python: 
```
$ export PATH="/home/username/anaconda2/bin:$PATH"
$ which python
/home/username/anaconda2/bin/python
```
Make changes in the `.bashrc` for permanent results.

* Install the required packages.
```
pip install mujoco-py==0.5.7
pip install theano
pip install gym
```   

* Clone the OpenAI-imitation (GAIL) repository.
`git clone https://github.com/openai/imitation.git`

* Clone the RAIL bitbucket repository.
`git clone https://abhisheknaik96@bitbucket.org/intelpclfad/rail.git`

* Add the path to the RAIL repository to `$PYTHONPATH`
* Copy the `expert_policies` directory from GAIL to RAIL.
* Run the initialization script:
```
username@machine:/path/to/RAIL$ chmod +x initialize.sh
username@machine:/path/to/RAIL$ ./initialize.sh
```
* Run the training script. Example usage:
```
username@machine:/path/to/RAIL$ chmod +x run/run_all/*
username@machine:/path/to/RAIL$ ./run/run_all/run_hopper.sh > training_logs/hopper_log.log
```

---

In case of any issues, feel free to contact the authors/developers :     

* Anirban Santara (nrbnsntr@gmail.com)    
* Abhishek Naik (abhisheknaik22296@gmail.com)

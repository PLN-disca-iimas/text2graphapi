#!/bin/bash

# >>>>> set params
LOG_INFO="# SH-LOG-INFO, $(date):"
echo "$LOG_INFO Shell script init"

# >>>>> set virtualenv
#conda activate /002/usuarios/andric.valdez/anaconda3/envs/text2graphapi/

# >>>>> exec scripts test:

# *** ISG: pan23 | pan22 | 20_News_Group | spanish_fake_news
#nohup python test.py -dn=pan23 -gt=isg -cpd=100 -sl=true -sg=true &
#nohup python test.py -dn=pan22 -gt=isg -cpd=100 -sl=true -sg=true &
#nohup python test.py -dn=20ng -gt=isg -cpd=100 -sl=true -sg=true &
#nohup python test.py -dn=spanish_fake_news -gt=isg -cpd=100 -sl=true -sg=true &
#nohup python test.py -dn=french_tgb -gt=isg -cpd=100 -sl=true -sg=true &

# *** Cooc: pan23 | pan22 | 20_News_Group | spanish_fake_news
#nohup python test.py -dn=pan23 -gt=cooccurrence -cpd=100 -sl=true -sg=true &
#nohup python test.py -dn=pan22 -gt=cooccurrence -cpd=100 -sl=true -sg=true &
#nohup python test.py -dn=20ng -gt=cooccurrence -cpd=100 -sl=true -sg=true &
#nohup python test.py -dn=spanish_fake_news -gt=cooccurrence -cpd=100 -sl=true -sg=true &
#nohup python test.py -dn=french_tgb -gt=cooccurrence -cpd=100 -sl=true -sg=true &

# *** Hetero: pan23 | pan22 | 20_News_Group | spanish_fake_news
#nohup python test.py -dn=pan23 -gt=heterogeneous -cpd=100 -sl=true -sg=true &
#nohup python test.py -dn=pan22 -gt=heterogeneous -cpd=100 -sl=true -sg=true &
#nohup python test.py -dn=20ng -gt=heterogeneous -cpd=100 -sl=true -sg=true &
#nohup python test.py -dn=spanish_fake_news -gt=heterogeneous -cpd=100 -sl=true -sg=true &
#nohup python test.py -dn=french_tgb -gt=heterogeneous -cpd=100 -sl=true -sg=true &

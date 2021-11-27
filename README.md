# AutoETER
EMNLP 2020 Findings: AutoETER: Automated Entity Type Representation for Knowledge Graph Embedding [Paper](https://www.aclweb.org/anthology/2020.findings-emnlp.105.pdf).

This is our source code and data for the paper:
>***Guanglin Niu***, Bo Li, Yongfei Zhang, Shiliang Pu and Jingyang Li. AutoETER: Automated Entity Type Representation for Knowledge Graph Embedding. EMNLP 2020 Findings, 2020, 1172–1181.

Author: Dr. Guanglin Niu (beihangngl at buaa.edu.cn)

## Introduction
To explore the type information for any KG, we develop a novel KGE framework with ***Auto***mated ***E***ntity ***T***yp***E*** ***R***epresentation (AutoETER), which learns the latent type embedding of each entity by regarding each relation as a translation operation between the types of two entities with a relation-aware attention mechanism. Particularly, our approach could model and infer all the relation patterns and complex relations. 

## Dataset
We provide four datasets: FB15K, FB15K237, WN18 and YAGO3-10. You can find and download all the datasets on [Onedrive](https://1drv.ms/u/s!AjhjEjaTE0SbbceogcmdwSu9ME?e=zfw6sN).

## Features
We develop a novel relation-aware entity type representation learning, and constrain the type embeddings via a relation-type sampling in codes/dataloader.py.
<br/>Each relation has two kinds of embeddings, one for entity-specific triples in the complex space and one for type-specific triples in the real space.

## Example to run the codes
### Train and Test on FB15k: 
    python codes/run.py --cuda --do_train --do_valid --do_test --data_path/FB15k --model AutoETER -n 128 -b 1024 -d 1000  -td 200 -g 22.0 -gt 7.0 -gp 5.0 -al1 0.1 -al2 0.5 -a 1.0 -lr 0.0001 --max_steps 150000 -save models/AutoETER_fb15k_1 -ps 16
    
## Acknowledge
    @inproceedings{Niu:AutoETER,
      author    = {Guanglin Niu and
                   Bo Li and
                   Yongfei Zhang and
                   Shiliang Pu and
                   Jingyang Li},
      title     = {AutoETER: Automated Entity Type Representation for Knowledge Graph Embedding},
      booktitle = {Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing: Findings},
      year      = {2020}
    }

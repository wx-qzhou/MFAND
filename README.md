# Illustration
![image](https://user-images.githubusercontent.com/37830460/235340773-b79a7917-892b-4374-af75-06404dfc3b8b.png)

Citing https://github.com/timxiao317/E2E-Name-Disambiguation.  
  
Implementation of the study proposed in the paper <a href="https://ieeexplore.ieee.org/document/9590332">Multiple Features Driven Author Name Disambiguation</a>
  
We have shared all the code related to this research, and those who are interested in using it may refer to the dataset demo we have provided. By processing the relevant data like the demo, the code can be executed successfully. 

# Datasets
The datasets used in this study are available from the following websites:
```
1) Aminer_data: https://static.aminer.cn/misc/na-data-kdd18.zip
2) WhoIswho_data: https://www.aminer.cn/billboard/whoiswho
```

# Main packages
networkx < 2.0  
torch < 1.8  
python <= 3.7.5  

# Usage

```python
python Create_Graph.py  
python make_walk.py  
python train_mi_macro.py  
```

# Citation
```bibtex
@INPROCEEDINGS{9590332,
  author={Zhou, Qian and Chen, Wei and Wang, Weiqing and Xu, Jiajie and Zhao, Lei},
  booktitle={2021 IEEE International Conference on Web Services (ICWS)}, 
  title={Multiple Features Driven Author Name Disambiguation}, 
  year={2021},
  volume={},
  number={},
  pages={506-515},
  doi={10.1109/ICWS53863.2021.00071}}

```

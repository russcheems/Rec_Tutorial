# Rec_Tutorial

数据集：暂时只用了u.data，结构为 uid iid rating timestamp
u.genre 类型
u.info  u.data中的用户数，电影数和评分数。
u.item  结构为movie id | movie title | release date | video release date | IMDb URL 

                   unknown | Action | Adventure | Animation | Children's | Comedy | Crime | Documentary | Drama | Fantasy | Film-Noir | Horror | Musical | Mystery | Romance | Sci-Fi | Thriller | War | Western
                   后面这些是类型的独热编码
u.occupation 职业
u.user 个人信息 结构为uid age gender occupation zip code


后面u1到u5相当于5折交叉验证
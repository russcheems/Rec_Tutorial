# Rec_Tutorial

> 暂时只用 `u.data` 文件。

- **u.data**：主评分数据，结构为：
  
  | user_id | item_id | rating | timestamp |
  | ------- | ------- | ------ | --------- |
  
- **u.genre**：电影类型列表
- **u.info**：包含用户数、电影数和评分数等统计信息
- **u.item**：电影信息，结构为：
  
  | movie_id | movie_title | release_date | video_release_date | IMDb_URL | unknown | Action | Adventure | Animation | Children's | Comedy | Crime | Documentary | Drama | Fantasy | Film-Noir | Horror | Musical | Mystery | Romance | Sci-Fi | Thriller | War | Western |
  | -------- | ----------- | ------------ | ------------------ | -------- | ------- | ------ | --------- | --------- | ---------- | ------ | ----- | ----------- | ----- | ------- | --------- | ------ | ------- | ------- | ------- | ------ | -------- | --- | ------- |
  
  后面各列为类型的独热编码

- **u.occupation**：职业列表
- **u.user**：用户信息，结构为：
  
  | user_id | age | gender | occupation | zip_code |
  | ------- | --- | ------ | ---------- | -------- |

- **u1.base/u1.test ... u5.base/u5.test**：5折交叉验证划分的数据子集


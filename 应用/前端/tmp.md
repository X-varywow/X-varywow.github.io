

## 说明

涉及技术栈：
- 前端
  - vite
  - nodejs
  - vue
  - vue-router
  - element-plus
  - element-plus/icons-vue
- 后端
  - flask
  - SQLAlchemy
  - pandas
  - flask_sqlalchemy

pnpm 更好的 nodejs 包管理

## 后端

（1）起一个 sql 服务

```sql
select count(*) from search_attention;
select count(*) from social_media_attention;
select count(*) from social_media_attitudes;
```

25080

SET FOREIGN_KEY_CHECKS = 1;

（2）flask 服务

```python
python app.py
```

port: 5000

## 前端

```bash
pnpm config set registry https://registry.npm.taobao.org

pnpm install

pnpm dev
```

https://dev.mysql.com/downloads/mysql/



### 界面结构

app.register_blueprint(search_attention_router)
app.register_blueprint(social_media_attention_router)
app.register_blueprint(social_media_attitudes_router)
app.register_blueprint(user_router)


from bilibili_api import search, sync
from bilibili_api.search import SearchObjectType

type =SearchObjectType("video")
print(sync(search.search_by_type("千里之外",search_type=type)))
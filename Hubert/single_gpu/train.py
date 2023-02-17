
import sys
import os
import logging

# 这里相当于是对 logging 进行全局的设置；然后因为我们在这个设置之后再对其他文件进行导入，因此其他文件内的 logging 也是遵循这个全局的设置的；
logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
# logger 相当于是每个文件自己单独的一个 logger，其可以再单独进行独特的设置，从而对默认的全局设置进行覆盖；
logger = logging.getLogger(__name__)















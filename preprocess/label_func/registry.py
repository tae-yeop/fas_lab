from typing import Callable, Dict

LABEL_REGISTRY: Dict[str, Callable] = {}

def register_label_function(name: str):
    """
    라벨 함수를 레지스트리에 등록하는 데코레이터
    name: 레지스트리에 등록할 때 사용할 문자열 키
    """
    def decorator(func: Callable) -> Callable:
        # func를 레지스트리에 등록
        LABEL_REGISTRY[name] = func
        
        # 원래 함수를 반환 => 다른 데코레이터와 혼합 사용 가능
        return func
    return decorator
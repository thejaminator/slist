
from slist import Slist

async def test_par_map_async_max_parallel_tqdm():
    async def func(x: int) -> int:
        # wait 2 s
        await asyncio.sleep(2)
        return x * 2

    result = await Slist([1, 2, 3]).par_map_async(func, max_par=1, tqdm=True)
    assert result == Slist([2, 4, 6])

if __name__ == '__main__':
    import asyncio
    asyncio.run(test_par_map_async_max_parallel_tqdm())


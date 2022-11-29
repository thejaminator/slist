def test_docs_chain_example():
    from slist import Slist

    result = (
        Slist([1, 2, 3])
        .repeat_until_size_or_raise(20)
        .grouped(2)
        .map(lambda inner_list: inner_list[0] + inner_list[1] if inner_list.length == 2 else inner_list[0])
        .flatten_option()
        .distinct_by(lambda x: x)
        .map(str)
        .reversed()
        .mk_string(sep=",")
    )
    assert result == "5,4,3"

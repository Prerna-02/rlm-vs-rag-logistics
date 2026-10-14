from rlm_logistics.eval import score


def test_label_uses_final_answer_line():
    assert score("label", "It mentions the product but...\nANSWER: delivery", "delivery") == 1.0
    assert score("label", "ANSWER: product", "delivery") == 0.0


def test_number_tolerance():
    assert score("number", "ANSWER: 69.5%", 68.0) == 1.0
    assert score("number", "ANSWER: 72", 68.0) == 0.0
    assert score("number", "no idea", 68.0) == 0.0


def test_pairs_f1_ignores_order_inside_pair():
    truth = [["O00001", "O00002"], ["O00003", "O00004"]]
    assert score("pairs", 'ANSWER: [["O00002", "O00001"], ["O00003", "O00004"]]', truth) == 1.0
    assert score("pairs", "ANSWER: [[O00001, O00002], [O00009, O00008]]", truth) == 0.5

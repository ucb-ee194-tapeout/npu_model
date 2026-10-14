"""DMA instruction words must match the spec table (npu_spec/06) and the RTL decoder
(Instructions.scala): dma.config is 7F/00, dma.wait 7F/01, load 7B/00, store 7B/01,
with the channel in funct3."""
import io

import pytest

from npu_model.util.converter import stream_to_instrs


def fields(word: int) -> tuple[int, int, int]:
    return word >> 25, (word >> 12) & 7, word & 0x7F


@pytest.mark.parametrize("channel", range(8))
def test_dma_words_follow_the_spec(channel: int):
    words = {name: stream_to_instrs(io.StringIO(text))[0].to_bytecode() & 0xFFFFFFFF for name, text in {
        "config": f"dma.config x5, {channel}",
        "wait": f"dma.wait {channel}",
        "load": f"dma.load x6, x10, x12, {channel}",
        "store": f"dma.store x3, x8, x12, {channel}",
    }.items()}
    assert fields(words["config"]) == (0b0000000, channel, 0b1111111)
    assert fields(words["wait"]) == (0b0000001, channel, 0b1111111)
    assert fields(words["load"]) == (0b0000000, channel, 0b1111011)
    assert fields(words["store"]) == (0b0000001, channel, 0b1111011)
    # The RTL's DMA_CONFIG_ANY / DMA_WAIT_ANY patterns also fix rd = 0 and rs2 = 0.
    assert (words["config"] >> 7) & 31 == 0 and (words["config"] >> 20) & 31 == 0
    assert (words["wait"] >> 7) & 31 == 0 and (words["wait"] >> 15) & 0x3FF == 0

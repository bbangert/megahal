defmodule MegahalMindTest do
  use ExUnit.Case
  doctest Megahal.Mind

  test "loads the brain file" do
    mind = Megahal.Mind.new()
    mind = Megahal.Mind.train_on_file(mind, "pulp.txt")
    {_mind, reply} = Megahal.Mind.reply(mind, "What happened to Yolanda?")
    IO.puts(inspect(reply))
    {_mind, reply} = Megahal.Mind.reply(mind, "What happened to Yolanda?")
    IO.puts(inspect(reply))
  end

  describe "decompose" do
    test "decomposes a sentence" do
      assert Megahal.Mind.decompose("Hello, world!") == {["", ", ", "!"], ["HELLO", "WORLD"], ["Hello", "world"]}
      assert Megahal.Mind.decompose("Isn't this neat?") == {["", " ", " ", "?"], ["ISN'T", "THIS", "NEAT"], ["Isn't", "this", "neat"]}
      assert Megahal.Mind.decompose(" I can't eat the hob-goblin!") == {[" ", " ", " ", " ", " ", "!"], ["I", "CAN'T", "EAT", "THE", "HOB-GOBLIN"], ["I", "can't", "eat", "the", "hob-goblin"]}
    end
  end

  describe "dictionary_lookup" do
    test "looks up values" do
      mind = Megahal.Mind.new()
      {mind, symbol} = Megahal.Mind.dictionary_lookup("hello", mind)
      assert inspect(mind) == "#Megahal.Mind<seed: 0, fore: 0, back: 0, case: 0, punc: 0, dictionary: 451, brain: 0>"
      assert symbol == 450
    end
  end
end

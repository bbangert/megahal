defmodule MegahalMindTest do
  use ExUnit.Case
  doctest Megahal.Mind

  test "loads the brain file" do
    _mind = Megahal.Mind.new()
    # mind = Megahal.Mind.train_on_file(mind, "pulp.txt")
    # {_mind, reply} = Megahal.Mind.reply(mind, "What happened to Yolanda?")
    # IO.puts(inspect(reply))
    # {_mind, reply} = Megahal.Mind.reply(mind, "What happened to Yolanda?")
    # IO.puts(inspect(reply))
  end
end

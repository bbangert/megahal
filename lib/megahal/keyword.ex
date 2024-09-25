defmodule Megahal.Keyword do
  @moduledoc """
  Helper functions to extract keywords from a list of words.
  """

  @swap_map Megahal.Constants.swap()
  @greeting Megahal.Constants.greeting()
  @banned Megahal.Constants.banned()

  def extract(nil), do: @greeting

  def extract([]), do: @greeting

  @spec extract(list(String.t())) :: [String.t()]
  @doc """
  Extracts keywords from a list of words, swaps antonyms, and removes banned words.

  Note that all words must be upper-cased.

  ## Examples

      iex> Megahal.Keyword.extract(["HATE", "WORLD"])
      ["LOVE", "WORLD"]

      iex> Megahal.Keyword.extract(["SORRY", "ABOUT", "ME", "TOO"])
      ["APOLOGY", "YOU"]

      iex> Megahal.Keyword.extract(["8492", "IS", "NEAT"])
      ["NEAT"]

  """
  def extract(words) do
    words
    |> Enum.map(&process_word/1)
    |> Enum.reject(&is_nil/1)
    |> Enum.uniq()
  end

  defp process_word(word) do
    cond do
      String.match?(word, ~r/^[0-9]/) -> nil
      is_map_key(@banned, word) -> nil
      true -> Map.get(@swap_map, word, word)
    end
  end
end

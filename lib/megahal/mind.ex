defmodule Megahal.Mind do
  @moduledoc """
  The MegaHAL mind is a stateful module that can be trained on a corpus of text
  and then used to generate replies based on the training.
  """
  use TypedStruct

  alias Sooth.Predictor
  alias __MODULE__, as: Mind

  @aux Megahal.Constants.auxiliary()

  typedstruct enforce: true do
    @typedoc """
    The state of the MegaHAL mind. This struct contains the five models that
    MegaHAL uses to generate replies. The `seed` model is used to start the
    forwards-backwards reply generation. The `fore` and `back` models are used
    to generate the middle of the reply. The `case` model is used to generate
    the case of the words in the reply. The `punc` model is used to generate
    the punctuation in the reply. The `dictionary` is used to map words to
    integers, and the `brain` is used to map bigrams to integers used as the
    context id for the Predictor.
    """
    field(:seed, Predictor.t())
    field(:fore, Predictor.t())
    field(:back, Predictor.t())
    field(:case, Predictor.t())
    field(:punc, Predictor.t())
    field(:dictionary, map())
    field(:brain, map())
  end

  @doc """
  Create a new MegaHAL mind with default values.
  """
  @spec new() :: Mind.t()
  def new() do
    mind = %Mind{
      seed: Predictor.new(0),
      fore: Predictor.new(0),
      back: Predictor.new(0),
      case: Predictor.new(0),
      punc: Predictor.new(0),
      dictionary: %{"<error>" => 0, "<fence>" => 1, "<blank>" => 2},
      brain: %{}
    }

    # Load all the constants into the dictionary
    mind
    |> init_dict(Map.keys(Megahal.Constants.auxiliary()))
    |> init_dict(Map.keys(Megahal.Constants.banned()))
    |> init_dict(Map.keys(Megahal.Constants.swap()))
    |> init_dict(Map.values(Megahal.Constants.swap()))
  end

  defp init_dict(mind, words) do
    Enum.reduce(words, mind, fn word, mind ->
      elem(dictionary_lookup(word, mind), 0)
    end)
  end

  @doc """
  Train a MegaHAL mind on a file of text. The file should be a UTF-8 encoded
  text file with one sentence per line.
  """
  @spec train_on_file(Mind.t(), String.t()) :: Mind.t()
  def train_on_file(mind, filename) do
    filename
    |> File.stream!([:trim_bom, encoding: :utf8])
    |> Stream.map(&String.trim/1)
    |> Enum.reduce(mind, &train(&2, &1))
  end

  @doc """
  Save the MegaHAL mind to a file.
  """
  @spec save(Mind.t(), String.t()) :: :ok
  def save(mind, filename) do
    File.write!(filename, :erlang.term_to_binary(mind), [:compressed])
  end

  @doc """
  Load the MegaHAL mind from a file.

  ## Parameters

    * `filename` - The name of the file to load the MegaHAL mind from.
  """
  @spec load(String.t()) :: {:ok, Mind.t()} | {:error, any()}
  def load(filename) do
    with {:ok, file} <- File.open(filename, [:read, :binary, :compressed]) do
      mind = :erlang.binary_to_term(IO.binread(file, :eof))
      File.close(file)
      {:ok, mind}
    end
  end

  @doc """
  Train the MegaHAL mind on a line of text.
  """
  @spec train(Mind.t(), String.t()) :: Mind.t()
  def train(mind, line), do: line |> String.trim() |> decompose() |> learn(mind)

  @doc """
  Generate a reply based on the MegaHAL mind's training.
  """
  @spec reply(Megahal.Mind.t(), String.t()) :: {Megahal.Mind.t(), nil | String.t()}
  def reply(mind, line) do
    {_puncs, norms, _words} = decompose(String.trim(line))

    {mind, keyword_symbols} =
      norms
      |> Megahal.Keyword.extract()
      |> Enum.reject(&is_nil/1)
      |> words_to_symbols(mind)

    {mind, input_symbols} = words_to_symbols(norms, mind)
    {mind, utterances} = generate_utterances(mind, keyword_symbols, [], 10)

    utterances
    |> Enum.reject(&(&1 == input_symbols or is_nil(&1)))
    |> then(&best_utterance(mind, &1, keyword_symbols))
  end

  @spec best_utterance(Mind.t(), [String.t()], [String.t()]) :: {Mind.t(), String.t() | nil}
  defp best_utterance(mind, utterances, keyword_symbols),
    do: best_utterance(mind, utterances, keyword_symbols, nil)

  @spec best_utterance(Mind.t(), [], any(), String.t() | nil) :: {Mind.t(), String.t() | nil}
  defp best_utterance(mind, [], _keyword_symbols, reply), do: {mind, reply}

  defp best_utterance(mind, utterances, keyword_symbols, reply) do
    with {:ok, mind, utterance} <- select_utterance(mind, utterances, keyword_symbols) do
      # Note that we can't collapse this any further because `best_utterance` needs the utterance
      case rewrite(mind, utterance) do
        # If the rewrite fails, remove the utterance from the list and try again
        {mind, nil} ->
          best_utterance(
            mind,
            Enum.reject(utterances, &(&1 == utterance)),
            keyword_symbols,
            reply
          )

        {mind, words} ->
          {mind, words}
      end
    else
      {:error, :no_utterance} -> {mind, reply}
    end
  end

  defp select_utterance(mind, utterances, keyword_symbols) do
    {mind, utterance, _} =
      Enum.reduce(utterances, {mind, nil, -1}, fn utterance, {mind, best_utterance, best_score} ->
        {mind, score} = calculate_score(mind, utterance, keyword_symbols)

        if(score > best_score,
          do: {mind, utterance, score},
          else: {mind, best_utterance, best_score}
        )
      end)

    cond do
      is_nil(utterance) -> {:error, :no_utterance}
      true -> {:ok, mind, utterance}
    end
  end

  defp calculate_score(mind, utterance, keyword_symbols) do
    {mind, score} = calculate_model_score(mind, mind.fore, utterance, keyword_symbols, 0, [1, 1])
    reversed = Enum.reverse(utterance)

    {mind, score} =
      calculate_model_score(mind, mind.back, reversed, keyword_symbols, score, [1, 1])

    score = if(length(utterance) > 7, do: score / Math.sqrt(length(utterance) - 1), else: score)
    score = if(length(utterance) > 15, do: score / length(utterance), else: score)
    {mind, score}
  end

  defp calculate_model_score(mind, _model, [], _keyword_symbols, score, _context),
    do: {mind, score}

  defp calculate_model_score(mind, model, [norm | rest], keyword_symbols, score, [first, second]) do
    {mind, id} = brain_lookup([first, second], mind)

    score =
      with true <- Enum.member?(keyword_symbols, norm),
           {:ok, surprise} <- Predictor.surprise(model, id, norm) do
        score + surprise
      else
        _ -> score
      end

    calculate_model_score(mind, model, rest, keyword_symbols, score, [second, norm])
  end

  @spec words_to_symbols([String.t()], Mind.t()) :: {Mind.t(), [non_neg_integer()]}
  defp words_to_symbols(words, mind) do
    Enum.reduce(words, {mind, []}, fn word, {mind, symbols} ->
      {mind, symbol} = dictionary_lookup(word, mind)
      {mind, symbols ++ [symbol]}
    end)
  end

  defp generate_utterances(mind, _keyword_symbols, utterances, 0), do: {mind, utterances}

  defp generate_utterances(mind, keyword_symbols, utterances, remaining) do
    {mind, utterance} = generate_reply(mind, keyword_symbols)
    generate_utterances(mind, keyword_symbols, utterances ++ [utterance], remaining - 1)
  end

  @spec rewrite(Mind.t(), [non_neg_integer()]) :: {Mind.t(), String.t() | nil}
  defp rewrite(mind, norm_symbols) do
    with {:ok, mind, word_symbols} <- norms_to_words(mind, norm_symbols, [], [1, 1], 10) do
      # We've used the case model to rewrite the norms to a words in a way that
      # guarantees that each adjacent pair of words has been previously observed.
      # Now we use the punc model to generate the word-separators to be inserted
      # between the words in the reply.
      decode_dict = Map.new(mind.dictionary, fn {k, v} -> {v, k} end)
      {mind, punc_symbols} = words_to_puncs(mind, [], [1] ++ word_symbols ++ [1])

      words =
        Enum.zip(punc_symbols, word_symbols ++ [nil])
        |> Enum.flat_map(fn {a, b} -> [a, b] end)
        |> Enum.reject(&is_nil/1)
        |> Enum.map(&Map.get(decode_dict, &1))
        |> Enum.join("")

      {mind, words}
    else
      _ -> {mind, nil}
    end
  end

  defp words_to_puncs(mind, punc_symbols, context) when length(context) < 2,
    do: {mind, punc_symbols}

  defp words_to_puncs(mind, punc_symbols, [prev, word | rest]) do
    {mind, id} = brain_lookup([prev, word], mind)
    limit = Predictor.count(mind.punc, id)
    punc_symbol = Predictor.select(mind.punc, id, Enum.random(1..limit))
    words_to_puncs(mind, punc_symbols ++ [punc_symbol], [word | rest])
  end

  # Exceeded retries
  defp norms_to_words(mind, _, _, _, 0), do: {:error, :out_of_retries, mind}

  # Got a full set of norm symbols, verify last one was seen or try again
  # We verify that the final word has been previously observed.
  defp norms_to_words(mind, norm_symbols, word_symbols, _context, retries)
       when length(norm_symbols) == length(word_symbols) do
    {mind, id} = brain_lookup([List.last(word_symbols), 1], mind)

    case Predictor.count(mind.punc, id) do
      count when count > 0 -> {mind, word_symbols}
      _ -> norms_to_words(mind, norm_symbols, [], [1 | norm_symbols], retries - 1)
    end

    {:ok, mind, word_symbols}
  end

  defp norms_to_words(mind, norm_symbols, word_symbols, [prev, norm | rest], retries) do
    {mind, id} = brain_lookup([prev, norm], mind)

    case Predictor.fetch_random_select(mind.case, id) do
      {:ok, word} ->
        norms_to_words(mind, norm_symbols, word_symbols ++ [word], [word | rest], retries)

      :error ->
        norms_to_words(mind, norm_symbols, [], [1 | norm_symbols], retries - 1)
    end
  end

  @spec generate_reply(Mind.t(), [String.t()]) :: {Mind.t(), [non_neg_integer()]}
  defp generate_reply(mind, keyword_symbols) do
    case select_keyword(mind, keyword_symbols) do
      nil -> random_walk(mind.fore, mind, [1, 1], keyword_symbols)
      keyword -> generate_reply(mind, keyword_symbols, keyword)
    end
  end

  defp generate_reply(mind, keyword_symbols, keyword) do
    contexts = [[2, keyword], [keyword, 2]]

    {contexts, mind} =
      Enum.reduce(contexts, {[], mind}, fn context, {contexts, mind} ->
        {mind, id} = brain_lookup(context, mind)

        case Predictor.fetch_random_select(mind.seed, id) do
          {:ok, ctx} ->
            index = Enum.find_index(context, &(&1 == 2))
            {contexts ++ List.replace_at(context, index, ctx), mind}

          _ ->
            {contexts, mind}
        end
      end)

    cond do
      length(contexts) > 0 ->
        context = Enum.random(contexts)
        glue = Enum.filter(context, &(&1 == 1))

        {mind, back_results} =
          random_walk(mind.back, mind, Enum.reverse(context), keyword_symbols)

        {mind, fore_results} = random_walk(mind.fore, mind, context, keyword_symbols)
        {mind, Enum.reverse(back_results) ++ glue ++ fore_results}

      true ->
        {mind, []}
    end
  end

  # Start a random walk with the rest of the arguments set to their defaults.
  # This is classic Markovian generation; using a model, start with a context
  # and continue until we hit a <fence> symbol. The only addition here is that
  # we roll the dice several times, and prefer generations that elicit a
  # keyword.
  defp random_walk(model, mind, context, keywords) do
    random_walk(model, mind, context, keywords, 0, [], 10)
  end

  # Stop and return no results if we hit an error (0) symbol
  defp random_walk(_model, mind, _context, _keys, 0, _results, 0), do: {mind, []}

  # Stop and return our results if we hit a <fence> (1) symbol
  defp random_walk(_model, mind, _context, _keys, 1, results, 0), do: {mind, results}

  # We got a symbol that isn't 0 or 1, and out of retries so move onto the next context
  defp random_walk(model, mind, [_first, second], keys, symbol, results, 0) do
    random_walk(model, mind, [second, symbol], keys, symbol, results ++ [symbol], 10)
  end

  defp random_walk(model, mind, context, keywords, _, results, times) do
    {mind, id} = brain_lookup(context, mind)

    case Predictor.fetch_random_select(model, id) do
      {:ok, symbol} ->
        random_walk(
          model,
          mind,
          context,
          Enum.filter(keywords, &(&1 == symbol)),
          symbol,
          results,
          # Break early by setting retries to 0 if we find a keyword
          if(symbol in keywords, do: 0, else: times - 1)
        )

      _ ->
        random_walk(model, mind, context, keywords, 0, results, times - 1)
    end
  end

  # Remove auxilliary words and select at random from what remains
  defp select_keyword(%Mind{dictionary: dictionary}, keyword_symbols) do
    aux_symbols =
      @aux
      |> Map.keys()
      |> Enum.map(&Map.get(dictionary, &1))
      |> Enum.filter(&is_nil/1)
      |> Map.new(&{&1, 0})

    case Enum.filter(keyword_symbols, &is_map_key(aux_symbols, &1)) do
      [] -> nil
      keywords -> Enum.random(keywords)
    end
  end

  defp learn({nil, nil, nil}, mind), do: mind

  # Train each of the five models based on a sentence decomposed into a list of
  # word separators (puncs), capitalised words (norms) and words as they were
  # observed (in mixed case).
  defp learn({puncs, norms, words}, mind) do
    # Convert the three lists of strings into three lists of symbols so that we
    # can use the Sooth.Predictor. This is done by finding the ID of each of
    # the strings in the :dictionary, allowing us to easily rewrite each symbol
    # back to a string later.
    {mind, punc_symbols} = Enum.reduce(puncs, {mind, []}, &lookup_dictionary_term/2)
    {mind, norm_symbols} = Enum.reduce(norms, {mind, []}, &lookup_dictionary_term/2)
    {mind, word_symbols} = Enum.reduce(words, {mind, []}, &lookup_dictionary_term/2)

    mind
    |> update_seed(norm_symbols ++ [1], 1)
    |> update_fore(norm_symbols, [1, 1])
    |> update_back(Enum.reverse(norm_symbols), [1, 1])
    |> update_case(Enum.zip(word_symbols, norm_symbols), 1)
    |> update_punc(Enum.zip(punc_symbols, word_symbols ++ [1]), 1)
  end

  defp lookup_dictionary_term(term, {mind, terms}),
    do: dictionary_lookup(term, mind) |> then(fn {mind, value} -> {mind, terms ++ [value]} end)

  # The :seed model is used to start the forwards-backwards reply generation.
  # Given a keyword, we want to find a word that has been observed adjacent to
  # it. Each context here is a bigram where one symbol is the keyword and the
  # other is the special <blank> symbol (which has ID 2). The model learns
  # which words can fill the blank.
  defp update_seed(mind, [], _), do: mind

  defp update_seed(mind, [norm | norm_symbols], prev) do
    {mind, id} =
      brain_lookup([prev, 2], mind)
      |> then(fn {mind, id} -> put_in(mind.seed, Predictor.observe(mind.seed, id, norm)) end)
      |> then(&brain_lookup([2, norm], &1))

    update_seed(%Mind{mind | seed: Predictor.observe(mind.seed, id, norm)}, norm_symbols, norm)
  end

  # The :fore model is a classic second-order Markov model that can be used to
  # generate an utterance in a random-walk fashion. For each adjacent pair of
  # symbols the model learns which symbols can come next. Note that the
  # special <fence> symbol (which has ID 1) is used to delimit the utterance.
  defp update_fore(mind, [], [first, second]) do
    {mind, id} = brain_lookup([first, second], mind)
    mind = put_in(mind.fore, Predictor.observe(mind.fore, id, 1))
    mind
  end

  defp update_fore(mind, [norm | norm_symbols], [first, second]) do
    {mind, id} = brain_lookup([first, second], mind)

    update_fore(put_in(mind.fore, Predictor.observe(mind.fore, id, norm)), norm_symbols, [
      second,
      norm
    ])
  end

  # The :back model is similar to the :fore model; it simply operates in the
  # opposite direction. This is how the original MegaHAL was able to generate
  # a random sentence guaranteed to contain a keyword; the :fore model filled
  # in the gaps towards the end of the sentence, and the back model filled in
  # the gaps towards the beginning of the sentence.
  defp update_back(mind, [], [first, second]) do
    {mind, id} = brain_lookup([first, second], mind)
    put_in(mind.back, Predictor.observe(mind.back, id, 1))
  end

  defp update_back(mind, [norm | norm_symbols], [first, second]) do
    {mind, id} = brain_lookup([first, second], mind)

    update_back(put_in(mind.back, Predictor.observe(mind.back, id, norm)), norm_symbols, [
      second,
      norm
    ])
  end

  # The previous three models were all learning the sequence of norms, which
  # are capitalised words. When we generate a reply, we want to rewrite it so
  # MegaHAL doesn't speak in ALL CAPS. The case model achieves this. For the
  # previous word and the current norm it learns what the next word should be.
  defp update_case(mind, [], _), do: mind

  defp update_case(mind, [{word, norm} | word_norms], first) do
    {mind, id} = brain_lookup([first, norm], mind)
    update_case(put_in(mind.case, Predictor.observe(mind.case, id, word)), word_norms, word)
  end

  # After generating a list of words, we need to join them together with
  # word-separators (whitespace and punctuation) in-between. The :punc model
  # is used to do this; here it learns for two adjacent words which
  # word-separators can be used to join them together.
  defp update_punc(mind, [], _), do: mind

  defp update_punc(mind, [{punc, word} | punc_words], first) do
    {mind, id} = brain_lookup([first, word], mind)
    update_punc(put_in(mind.punc, Predictor.observe(mind.punc, id, punc)), punc_words, word)
  end

  # Given a term, look up its ID in the brain. If it doesn't exist, add it.
  @spec brain_lookup(term :: list(), mind :: Mind.t()) :: {Mind.t(), integer()}
  defp brain_lookup(term, %Mind{brain: brain} = mind) do
    case Map.get(brain, term) do
      nil -> brain |> map_size() |> then(&{put_in(mind.brain, Map.put(brain, term, &1)), &1})
      value -> {mind, value}
    end
  end

  @doc false
  @spec dictionary_lookup(word :: String.t(), mind :: Mind.t()) :: {Mind.t(), integer()}
  def dictionary_lookup(word, %Mind{dictionary: dict} = mind) do
    case Map.get(dict, word) do
      nil -> dict |> map_size() |> then(&{put_in(mind.dictionary, Map.put(dict, word, &1)), &1})
      value -> {mind, value}
    end
  end

  @doc false
  @spec decompose(String.t()) :: {nil, nil, nil} | {[String.t()], [String.t()], String.t()}
  def decompose(""), do: {nil, nil, nil}

  def decompose(line) do
    {punc, words} = segment(line)
    {punc, Enum.map(words, &String.upcase/1), words}
  end

  # This segments a sentence into two arrays representing word-separators and
  # the original words themselves.
  @spec segment(String.t()) :: {list(String.t()), list(String.t())}
  defp segment(line) do
    # split the sentence into an array of alternating words and word-separators
    Regex.split(~r/[[:word:]]+/u, line, include_captures: true)
    # ensure the array starts with and ends with a word-separator, even if it's the blank one
    |> then(&if(List.last(&1) =~ ~r/[[:word:]]+/u, do: &1 ++ [""], else: &1))
    |> then(&if(List.first(&1) =~ ~r/[[:word:]]+/u, do: [""] ++ &1, else: &1))
    # join all the candidate trigrams
    |> join_trigrams()
    # chunk the array into two-element lists, then reduce them into two lists
    |> Enum.chunk_every(2)
    |> Enum.reduce({[], []}, fn
      [punc, word], {puncs, words} -> {puncs ++ [punc], words ++ [word]}
      [punc], {puncs, words} -> {puncs ++ [punc], words}
    end)
  end

  # join trigrams of word-separator-word if the separator is a single ' or -
  # this means "don't" and "hob-goblin" become single words
  # implemented as a recursive function using a guard to check if the separator is ' or -
  defp join_trigrams([word1, separator, word2 | rest]) when separator in ["'", "-"] do
    [Enum.join([word1, separator, word2]) | join_trigrams(rest)]
  end

  defp join_trigrams([]), do: []
  defp join_trigrams([word | rest]), do: [word | join_trigrams(rest)]
end

defimpl Inspect, for: Megahal.Mind do
  def inspect(
        %Megahal.Mind{
          seed: seed,
          fore: fore,
          back: back,
          case: case_m,
          punc: punc,
          dictionary: dictionary,
          brain: brain
        },
        _opts
      ) do
    seed = map_size(seed.context_map)
    fore = map_size(fore.context_map)
    back = map_size(back.context_map)
    case_m = map_size(case_m.context_map)
    punc = map_size(punc.context_map)
    dictionary = map_size(dictionary)
    brain = map_size(brain)

    "#Megahal.Mind<seed: #{seed}, fore: #{fore}, back: #{back}, case: #{case_m}, punc: #{punc}, dictionary: #{dictionary}, brain: #{brain}>"
  end
end

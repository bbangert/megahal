defmodule Megahal.Personality do
  use GenServer

  def start_link() do
    GenServer.start_link(__MODULE__, [])
  end

  def load_file(pid, filename) do
    GenServer.cast(pid, {:load_file, filename})
  end

  def reply(pid, words) do
    GenServer.call(pid, {:reply, words})
  end

  def reply_and_learn(pid, words) do
    GenServer.call(pid, {:reply_and_learn, words})
  end

  @impl true
  def init(_args) do
    {:ok, Megahal.Mind.new()}
  end

  @impl true
  def handle_call({:reply, words}, _from, mind) do
    {mind, reply} = Megahal.Mind.reply(mind, words)
    {:reply, reply, mind}
  end

  @impl true
  def handle_call({:reply_and_learn, words}, _from, mind) do
    {mind, reply} = Megahal.Mind.reply(mind, words)
    GenServer.cast(self(), {:learn, words})
    {:reply, reply, mind}
  end

  @impl true
  def handle_cast({:learn, words}, mind) do
    mind = Megahal.Mind.train(mind, words)
    {:noreply, mind}
  end

  @impl true
  def handle_cast({:load_file, filename}, mind) do
    mind = Megahal.Mind.train_on_file(mind, filename)
    {:noreply, mind}
  end
end

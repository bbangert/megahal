defmodule Megahal.MixProject do
  use Mix.Project

  @version "0.1.0"
  @description "Meghal is a port of the MegaHAL chatbot to Elixir."

  def project do
    [
      app: :megahal,
      version: @version,
      name: "Megahal",
      description: @description,
      elixir: "~> 1.17",
      start_permanent: Mix.env() == :prod,
      deps: deps(),
      package: package()
    ]
  end

  # Run "mix help compile.app" to learn about applications.
  def application do
    [
      extra_applications: [:logger]
    ]
  end

  # Run "mix help deps" to learn about dependencies.
  defp deps do
    [
      {:typed_struct, "~> 0.3.0"},
      {:sooth, "~> 0.3.2"},
      {:credo, "~> 1.7", only: [:dev, :test], runtime: false},

      # Dev/test dependencies
      {:stream_data, "~> 1.1", only: :test},
      {:excoveralls, "~> 0.18.2", only: :test},
      {:ex_doc, "~> 0.34.2", only: :dev, runtime: false}
    ]
  end

  defp package do
    %{
      licenses: ["Unlicense"],
      maintainers: ["Ben Bangert"],
      links: %{"GitHub" => "https://github.com/bbangert/sooth"}
    }
  end
end

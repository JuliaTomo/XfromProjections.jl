#push!(LOAD_PATH,"../src/")

using Documenter, XfromProjections

makedocs(sitename="Tomo document test",
    modules = [XfromProjections],
    pages = Any[
    "Home" => "index.md",
    "Tutorials" => Any[
      "tutorials/test.md"
    ]
    ]
)

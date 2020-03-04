
files = list.files("./",pattern = "*.R")
files = files[files!="main.R" ]

for(f in files){
  source(f)
}

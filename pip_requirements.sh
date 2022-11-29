pip_requirements() {
if test "$#" -eq 0
then 
  echo $'\nProvide at least one Python package name\n' 
else 
  for package in "$@"
  do
    pip3 install $package
    pip3 freeze | grep -i $package >> requirements.txt
  done
fi
}
function trainpredict() {
  python train.py && python a.py
}

function Label_Error() {
  return 1
}

trainpredict
if [ $? = 1 ]
then
	./train-and-predict.sh
fi

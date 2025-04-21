package tensor

import (
	"os"

	"gopkg.in/yaml.v2"
)

func LoadShape(filePath string) (shape Shape, err error) {
	var shapeData []byte
	shapeData, err = os.ReadFile(filePath + ".shape")
	if err != nil {
		return
	}

	err = yaml.Unmarshal(shapeData, &shape)
	if err != nil {
		return
	}
	return
}
func LoadTensor(filePath string) (tensor Tensor, err error) {
	var data []byte

	data, err = os.ReadFile(filePath + ".data")
	if err != nil {
		return
	}
	_, err = os.ReadFile(filePath + ".shape")
	if err != nil {
		return
	}
	var shape Shape
	shape, err = LoadShape(filePath)
	if err != nil {
		return
	}
	tensor = Tensor{Data: data, Shape: shape}
	return
}

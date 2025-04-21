package tensor

import (
	"encoding/binary"
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
func LoadTensor[T Number](filePath string) (tensor Tensor[T], err error) {

	_, err = os.ReadFile(filePath + ".shape")
	if err != nil {
		return
	}
	var shape Shape
	shape, err = LoadShape(filePath)
	if err != nil {
		return
	}
	file, err := os.Open(filePath + ".data")
	if err != nil {
		return
	}
	defer file.Close()
	data := make([]T, shape.Size)

	err = binary.Read(file, binary.LittleEndian, data)
	if err != nil {
		return
	}
	tensor = Tensor[T]{Data: data, Shape: shape}
	return
}

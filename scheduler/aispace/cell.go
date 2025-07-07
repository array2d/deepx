package aispace

type CellValue any
type Cell struct {
	// Key 唯一标识符
	Key string `json:"key"`
	// Value 存储的值，可以是任意类型
	Value CellValue `json:"value"`
	// Type 类型标识符，便于区分不同类型的 Cell
}

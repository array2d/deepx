package space

// Space 定义了通用的空间访问接口
// get/set/mv/del 分别用于获取、设置、移动、删除对象
// key 为对象标识，value 为任意类型

type Space interface {
	// Get 根据 key 获取对象，返回对象和是否存在
	Get(key string) (interface{}, bool)
	// Set 设置 key 对应的对象
	Set(key string, value interface{})
	// Mv 移动/重命名对象，返回是否成功
	Mv(srcKey, dstKey string) bool
	// Del 删除对象，返回是否成功
	Del(key string) bool
}

func GetSpace() Space {
	// 这里可以根据需要返回具体的 Space 实现
	// 例如，可以返回一个内存空间实现或分布式空间实现
	return nil // 返回 nil 仅为示例，实际应返回具体实现
}

package space

import (
	"context"

	"github.com/go-redis/redis/v8"
)

type RedisSpace struct {
	// Redis 客户端连接
	client *redis.Client
}

func NewRedisSpace(addr string, password string, db int) *RedisSpace {
	client := redis.NewClient(&redis.Options{
		Addr:     addr,
		Password: password, // no password set
		DB:       db,       // use default DB
	})

	return &RedisSpace{client: client}
}
func (r *RedisSpace) Get(key string) (interface{}, bool) {
	val, err := r.client.Get(context.Background(), key).Result()
	if err == redis.Nil {
		return nil, false // key does not exist
	} else if err != nil {
		return nil, false // error occurred
	}
	return val, true // return value and true for existence
}
func (r *RedisSpace) Set(key string, value interface{}) {
	err := r.client.Set(context.Background(), key, value, 0).Err()
	if err != nil {
		// handle error, e.g., log it
	}
}
func (r *RedisSpace) Mv(srcKey, dstKey string) bool {
	err := r.client.Rename(context.Background(), srcKey, dstKey).Err()
	if err == redis.Nil {
		return false // source key does not exist
	} else if err != nil {
		return false // error occurred
	}
	return true // successful rename
}
func (r *RedisSpace) Del(key string) bool {
	err := r.client.Del(context.Background(), key).Err()
	if err == redis.Nil {
		return false // key does not exist
	} else if err != nil {
		return false // error occurred
	}
	return true // successful deletion
}

参考资料：
- [菜鸟教程-GO](https://www.runoob.com/go/go-tutorial.html)
- [【后端专场 学习资料一】第五届字节跳动青训营](https://juejin.cn/post/7188225875211452476)
- https://go.dev/learn/
- [Go语言圣经（中文版）](https://books.studygolang.com/gopl-zh/)
- https://go.nsddd.top/
- [go 语言学习路线](https://bytedance.feishu.cn/docs/doccn3SFTuFIAVr4CDZGx48KKdd)
- [《Effective Go》](https://github.com/bingohuang/effective-go-zh-en)


## hello world

```go
package main
import "fmt"
func main(){
	fmt.Println("hello, world")
}
```

```bash
go version
#>>> go version go1.19.4 windows/amd64

go run hello.go
#>>> hello, world
```

还可以使用 build 来构建项目

```bash
go build hello.go

hello
#>>> hello, world
```

之后会生成一个 exe 可执行文件，1.86mb。

-----------------------

（注释版）

```go
package main
//指明文件属于哪个包，main 表示一个可独立执行的程序，每个 Go应用程序都应包含一个

import "fmt"
//需要使用 fmt 包（格式化 IO）

func main(){
// main 函数是每一个可执行程序所必须包含的
// 左括号不能单独放一行
    fmt.Println("hello,world")
}
```


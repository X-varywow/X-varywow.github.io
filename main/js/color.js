
// 利用 input 实现复制功能
function copyContent(ElementObj) {
    //var clickContent = grbToHex(ElementObj.style.background);
    var clickContent = ElementObj.id;
    console.log("已复制：", ElementObj.id)
    var inputElement = document.getElementById("copy_content");
    inputElement.value = clickContent;
    inputElement.select();
    document.execCommand("Copy");
}

// json 写入 html
function jsonTohtml(d) {
    //console.log(d);
    d.forEach(e => {    //json -> html
        c2 = ""
        e.children.forEach(colorinfo => {
            c2 += `<div class="card-wrap">
                <div class="card-color" onclick="copyContent(this)" id="${colorinfo.color}" style="background:${colorinfo.color}"></div>
                <div class="card-info">
                    <span class="hex">${colorinfo.info}</span>
                </div>
                </div>` });
        c1 = `<h1 id="${e.head}" style="background:${e.children[2].color}">${e.head}</h1>`

        card = `<div class="head-wrap">${c1}${c2}</div>`
        $(".container").append(card);

        side_str = `<a href="#${e.head}">
            <div class="btn-wrap" style="background:${e.children[2].color}">
                <p>${e.head}</p>
            </div>
        </a>`
        $(".side-wrap").append(side_str);
    });
}

// jquery
$.getJSON("color.json", function (d) {
    jsonTohtml(d);
})

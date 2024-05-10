

## _preface_

做个有些复杂有意思的， 只有一个打斗场景；完成 [使用Unity实现动作游戏的打击感](https://www.bilibili.com/video/BV1fX4y1G7tv)


小结：
- 实现多段连招，动画器配合脚本中的连击数重置状态
- 在合适的动画帧中，激活子物体，并添加碰撞范围
- 打击感：顿帧，震动视角，动画等，符合预期并适度夸张

</br>

## _相关流程_

(1) 下载 [github文件](https://github.com/RedFF0000/AttackSense)

拖动 unitypackage 文件到 assets 中

</br>

(2) window -> animation -> animator

</br>

(3) 脚本文件

- PlayerController.cs 挂载在 player 上，提供参数及状态控制
- Enemy.cs 挂载在 enemy 上，hit 的控制
- AttackSense.cs 挂载在 camera 上，镜头震动













</br>

## _附录：代码_

```cs
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class AttackSense : MonoBehaviour
{
    private static AttackSense instance;
    public static AttackSense Instance
    {
        get
        {
            if (instance == null)
                instance = Transform.FindObjectOfType<AttackSense>();
            return instance;
        }
    }
    private bool isShake;

    // 协程进行顿帧
    public void HitPause(int duration)
    {
        StartCoroutine(Pause(duration));
    }

    IEnumerator Pause(int duration)
    {
        float pauseTime = duration / 60f;
        Time.timeScale = 0;
        yield return new WaitForSecondsRealtime(pauseTime);
        Time.timeScale = 1;
    }

    // 协程进行相机震动
    public void CameraShake(float duration, float strength)
    {
        if (!isShake)
            StartCoroutine(Shake(duration, strength));
    }

    IEnumerator Shake(float duration, float strength)
    {
        isShake = true;
        Transform camera = Camera.main.transform;
        Vector3 startPosition = camera.position;

        while (duration > 0)
        {
            camera.position = Random.insideUnitSphere * strength + startPosition;
            duration -= Time.deltaTime;
            yield return null;
        }
        // 还原相机震动 
        camera.position = startPosition;
        isShake = false;
    }
}
```



```cs
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class PlayerController : MonoBehaviour
{
    [Header("补偿速度")]
    public float lightSpeed;
    public float heavySpeed;
    [Header("打击感")]
    public float shakeTime;
    public int lightPause;
    public float lightStrength;
    public int heavyPause;
    public float heavyStrength;

    [Space]
    public float interval = 2f;
    private float timer;
    private bool isAttack;
    private string attackType;
    private int comboStep;

    public float moveSpeed;
    public float jumpForce;
    new private Rigidbody2D rigidbody;
    private Animator animator;
    private float input;
    private bool isGround;
    [SerializeField] private LayerMask layer;

    [SerializeField] private Vector3 check;

    void Start()
    {
        rigidbody = GetComponent<Rigidbody2D>();
        animator = GetComponent<Animator>();
    }

    void Update()
    {
        input = Input.GetAxisRaw("Horizontal");
        isGround = Physics2D.OverlapCircle(transform.position + new Vector3(check.x, check.y, 0), check.z, layer);

        animator.SetFloat("Horizontal", rigidbody.velocity.x);
        animator.SetFloat("Vertical", rigidbody.velocity.y);
        animator.SetBool("isGround", isGround);

        Move();
        Attack();
    }

    void Move()
    {
        if (!isAttack)
            rigidbody.velocity = new Vector2(input * moveSpeed, rigidbody.velocity.y);
        else
        {
            if (attackType == "Light")
                rigidbody.velocity = new Vector2(transform.localScale.x * lightSpeed, rigidbody.velocity.y);
            else if (attackType == "Heavy")
                rigidbody.velocity = new Vector2(transform.localScale.x * heavySpeed, rigidbody.velocity.y);
        }

        if (Input.GetButtonDown("Jump") && isGround)
        {
            rigidbody.velocity = new Vector2(0, jumpForce);
            animator.SetTrigger("Jump");
        }

        if (rigidbody.velocity.x < 0)
            transform.localScale = new Vector3(-1, 1, 1);
        else if (rigidbody.velocity.x > 0)
            transform.localScale = new Vector3(1, 1, 1);
    }

    void Attack()
    {
        if (Input.GetKeyDown(KeyCode.Return) && !isAttack)
        {
            isAttack = true;
            attackType = "Light";
            comboStep++;
            if (comboStep > 3)
                comboStep = 1;
            timer = interval;
            animator.SetTrigger("LightAttack");
            animator.SetInteger("ComboStep", comboStep);
        }
        if (Input.GetKeyDown(KeyCode.RightShift) && !isAttack)
        {
            isAttack = true;
            attackType = "Heavy";
            comboStep++;
            if (comboStep > 3)
                comboStep = 1;
            timer = interval;
            animator.SetTrigger("HeavyAttack");
            animator.SetInteger("ComboStep", comboStep);
        }


        if (timer != 0)
        {
            timer -= Time.deltaTime;
            if (timer <= 0)
            {
                timer = 0;
                comboStep = 0;
            }
        }
    }

    public void AttackOver()
    {
        isAttack = false;
    }

    private void OnTriggerEnter2D(Collider2D other)
    {
        if (other.CompareTag("Enemy"))
        {
            if (attackType == "Light")
            {
                AttackSense.Instance.HitPause(lightPause);
                AttackSense.Instance.CameraShake(shakeTime, lightStrength);
            }
            else if (attackType == "Heavy")
            {
                AttackSense.Instance.HitPause(heavyPause);
                AttackSense.Instance.CameraShake(shakeTime, heavyStrength);
            }

            if (transform.localScale.x > 0)
                other.GetComponent<Enemy>().GetHit(Vector2.right);
            else if (transform.localScale.x < 0)
                other.GetComponent<Enemy>().GetHit(Vector2.left);
        }
    }

    private void OnDrawGizmos()
    {
        Gizmos.DrawWireSphere(transform.position + new Vector3(check.x, check.y, 0), check.z);
    }
}
```

```cs
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Enemy : MonoBehaviour
{
    public float speed;
    private Vector2 direction;
    private bool isHit;
    private AnimatorStateInfo info;

    private Animator animator;
    private Animator hitAnimator;
    new private Rigidbody2D rigidbody;

    void Start()
    {
        animator = transform.GetComponent<Animator>();
        hitAnimator = transform.GetChild(0).GetComponent<Animator>();
        rigidbody = transform.GetComponent<Rigidbody2D>();
    }

    void Update()
    {
        info = animator.GetCurrentAnimatorStateInfo(0);
        if (isHit)
        {
            rigidbody.velocity = direction * speed;
            if (info.normalizedTime >= .6f)
                isHit = false;
        }
    }

    public void GetHit(Vector2 direction)
    {
        transform.localScale = new Vector3(-direction.x, 1, 1);
        isHit = true;
        this.direction = direction; ;
        animator.SetTrigger("Hit");
        hitAnimator.SetTrigger("Hit");
    }
}
```



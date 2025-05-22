using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class CubeMove : MonoBehaviour
{
    public Vector3 direction = Vector3.forward;
    public float speed = 1f; //hız

    public bool destroyAfterTime = true; //yol olma acma kapama
    public float lifetime = 15f; // hayatta kalma süresi
    private float creationTime;

    void Start()
    {
        GetComponent<Renderer>().material.color = RandomColor();
        creationTime = Time.time;
        direction = direction.normalized;
    }

    void Update()
    {
        transform.Translate(direction * speed * Time.deltaTime);

        //belirli süre sonra yok olma
        if (destroyAfterTime && Time.time > creationTime + lifetime)
        {
            Destroy(gameObject);
            return;
        }
    }
    private void OnTriggerEnter(Collider other)
    {
        if (other.gameObject.CompareTag("Selecter"))
        {
            if (gameObject.GetComponent<Renderer>().material.color == Color.white)
            {
                direction = Vector3.right * -1f;
            }
            else
            {
                direction = Vector3.right;
            }
        }
    }
    Color RandomColor()
    { 
        float rand = Random.value;

        if (rand < 0.7f) return Color.white;
        else if (rand < 0.85f) return Color.green;
        else return Color.blue;
    }
}

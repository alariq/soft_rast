typedef void (*FB_DrawCB_t)(void* puserdata);

struct Framebuffer* FB_Create(int w, int h);
void FB_Destroy(struct Framebuffer* fb);
void FB_Draw(const struct Framebuffer* fb, FB_DrawCB_t cb);


typedef void (*DT_UpdateCB_t)(void* dst, void* puserdata);
struct DynamicTexture* DT_Create(int w, int h);
void DT_Update(const struct DynamicTexture* dt, void* data, int size);
void DT_Update(const struct DynamicTexture* dt, void* puserdata, DT_UpdateCB_t cb);
unsigned int DT_GetTextureID(DynamicTexture* dt);
unsigned int DT_GetWidth(DynamicTexture* dt);
unsigned int DT_GetHeight(DynamicTexture* dt);
void DT_Destroy(struct DynamicTexture* dt);

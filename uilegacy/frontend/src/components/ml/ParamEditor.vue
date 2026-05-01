<!-- frontend/src/components/ml/ParamEditor.vue -->
<template>
  <div class="param-editor">
    <el-form label-position="top" size="small">
      <div v-for="param in schema" :key="param.name" class="param-item">
        <el-form-item :label="param.label || param.name">
          <template #label>
            <div class="label-with-help">
              <span>{{ param.label || param.name }}</span>
              <el-tooltip v-if="param.help" :content="param.help" placement="top">
                <el-icon class="help-icon"><QuestionFilled /></el-icon>
              </el-tooltip>
            </div>
          </template>

          <!-- Boolean -> Switch -->
          <el-switch 
            v-if="param.type === 'bool'" 
            v-model="modelValue[param.name]" 
          />

          <!-- Choice -> Select -->
          <el-select 
            v-else-if="param.type === 'choice' || param.choices" 
            v-model="modelValue[param.name]" 
            class="w-full"
          >
            <el-option 
              v-for="opt in param.choices" 
              :key="opt" 
              :label="opt" 
              :value="opt" 
            />
          </el-select>

          <!-- Number with Range -> Slider or InputNumber -->
          <div v-else-if="param.type === 'int' || param.type === 'float'" class="flex items-center gap-4">
            <el-slider 
              v-if="param.min !== undefined && param.max !== undefined"
              v-model="modelValue[param.name]" 
              :min="param.min" 
              :max="param.max" 
              :step="param.type === 'int' ? 1 : 0.01"
              class="flex-grow"
            />
            <el-input-number 
              v-model="modelValue[param.name]" 
              :min="param.min" 
              :max="param.max" 
              :step="param.type === 'int' ? 1 : 0.01"
              controls-position="right"
            />
          </div>

          <!-- Default -> Input -->
          <el-input 
            v-else 
            v-model="modelValue[param.name]" 
          />
        </el-form-item>
      </div>
    </el-form>
  </div>
</template>

<script setup lang="ts">
import { QuestionFilled } from '@element-plus/icons-vue'

interface ParamSpec {
  name: string
  label?: string
  type: string
  default: any
  help?: string
  choices?: any[]
  min?: number
  max?: number
}

const props = defineProps<{
  schema: ParamSpec[]
  modelValue: Record<string, any>
}>()

const emit = defineEmits(['update:modelValue'])
</script>

<style scoped>
.param-item {
  margin-bottom: 12px;
}
.label-with-help {
  display: flex;
  align-items: center;
  gap: 4px;
  font-size: 0.85rem;
  font-weight: 600;
  color: #606266;
}
.help-icon {
  font-size: 14px;
  color: #909399;
  cursor: help;
}
.w-full { width: 100%; }
</style>
